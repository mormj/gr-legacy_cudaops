/* -*- c++ -*- */
/*
 * Copyright 2010,2012,2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <legacy_cudaops/fft_filter.h>
#include <gnuradio/logger.h>
#include <volk/volk.h>
#include <cstring>
#include <iostream>
#include <memory>

#include "helper_cuda.h"

namespace gr {
namespace legacy_cudaops {
namespace kernel {

#define VERBOSE 0

/**************************************************************/
fft_filter_ccf::fft_filter_ccf(int decimation,
                               const std::vector<float>& taps,
                               int nthreads)
    : d_fftsize(-1), d_decimation(decimation), d_nthreads(nthreads)
{
    gr::configure_default_loggers(d_logger, d_debug_logger, "fft_filter_ccf");
    set_taps(taps);


}

/*
 * determines d_ntaps, d_nsamples, d_fftsize, d_xformed_taps
 */
int fft_filter_ccf::set_taps(const std::vector<float>& taps)
{
    int i = 0;
    d_taps = taps;
    compute_sizes(taps.size());

    d_tail.resize(tailsize());
    for (i = 0; i < tailsize(); i++)
        d_tail[i] = 0;

    gr_complex* in = d_fwdfft->get_inbuf();
    gr_complex* out = d_fwdfft->get_outbuf();

    float scale = 1.0 / d_fftsize;

    // Compute forward xform of taps.
    // Copy taps into first ntaps slots, then pad with zeros
    for (i = 0; i < d_ntaps; i++)
        in[i] = gr_complex(taps[i] * scale, 0.0f);

    for (; i < d_fftsize; i++)
        in[i] = gr_complex(0.0f, 0.0f);

    d_fwdfft->execute(); // do the xform

    // now copy output to d_xformed_taps
    for (i = 0; i < d_fftsize; i++)
        d_xformed_taps[i] = out[i];

    return d_nsamples;
}

// determine and set d_ntaps, d_nsamples, d_fftsize
void fft_filter_ccf::compute_sizes(int ntaps)
{
    int old_fftsize = d_fftsize;
    d_ntaps = ntaps;
    d_fftsize = (int)(2 * pow(2.0, ceil(log(double(ntaps)) / log(2.0))));
    d_nsamples = d_fftsize - d_ntaps + 1;

    if (VERBOSE) {
        std::ostringstream msg;
        msg << "fft_filter_ccf: ntaps = " << d_ntaps << " fftsize = " << d_fftsize
            << " nsamples = " << d_nsamples;
        GR_LOG_ALERT(d_logger, msg.str());
    }

    // compute new plans
    if (d_fftsize != old_fftsize) {
        d_fwdfft = std::make_unique<fft::fft_complex_fwd>(d_fftsize, d_nthreads);
        d_invfft = std::make_unique<fft::fft_complex_rev>(d_fftsize, d_nthreads);

        checkCudaErrors(cufftCreate(&d_plan_fwd));
        checkCudaErrors(cufftCreate(&d_plan_rev));

        size_t workSize;
        checkCudaErrors(cufftMakePlanMany(
            d_plan_fwd, 1, &d_fftsize, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch_size, &workSize));
        checkCudaErrors(cufftMakePlanMany(
            d_plan_rev, 1, &d_fftsize, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch_size, &workSize));

        checkCudaErrors(cudaMalloc((void**)&d_data_fwd,
                            sizeof(cufftComplex) * d_fftsize * d_batch_size));
        checkCudaErrors(cudaMalloc((void**)&d_data_rev,
                            sizeof(cufftComplex) * d_fftsize * d_batch_size));

        d_xformed_taps.resize(d_fftsize);

        checkCudaErrors(cudaMalloc((void**)&d_dev_taps,
                            sizeof(cufftReal) * d_fftsize * d_batch_size));

    }
}


std::vector<float> fft_filter_ccf::taps() const { return d_taps; }

unsigned int fft_filter_ccf::ntaps() const { return d_ntaps; }

unsigned int fft_filter_ccf::filtersize() const { return d_fftsize; }

int fft_filter_ccf::filter(int nitems, const gr_complex* input, gr_complex* output)
{
    int dec_ctr = 0;
    int j = 0;
    int ninput_items = nitems * d_decimation;

    for (int i = 0; i < ninput_items; i += d_nsamples) {

        #if 0
        memcpy(d_fwdfft->get_inbuf(), &input[i], d_nsamples * sizeof(gr_complex));

        for (j = d_nsamples; j < d_fftsize; j++)
            d_fwdfft->get_inbuf()[j] = 0;

        d_fwdfft->execute(); // compute fwd xform

        gr_complex* a = d_fwdfft->get_outbuf();
        gr_complex* b = d_xformed_taps.data();
        gr_complex* c = d_invfft->get_inbuf();

        volk_32fc_x2_multiply_32fc_a(c, a, b, d_fftsize);

        d_invfft->execute(); // compute inv xform

        #endif

        checkCudaErrors(
                cudaMemcpy(d_data_fwd, &input[i], d_nsamples * sizeof(gr_complex), cudaMemcpyHostToDevice));

        checkCudaErrors(
                cudaMemset(d_data_fwd + d_nsamples, 0, d_fftsize-d_nsamples));

        checkCudaErrors(cufftExecC2C(d_plan_fwd, d_data, d_data, CUFFT_FORWARD));


        // add in the overlapping tail

        for (j = 0; j < tailsize(); j++)
            d_invfft->get_outbuf()[j] += d_tail[j];

        // copy nsamples to output
        j = dec_ctr;
        while (j < d_nsamples) {
            *output++ = d_invfft->get_outbuf()[j];
            j += d_decimation;
        }
        dec_ctr = (j - d_nsamples);

        // stash the tail
        if (!d_tail.empty()) {
            memcpy(&d_tail[0],
                   d_invfft->get_outbuf() + d_nsamples,
                   tailsize() * sizeof(gr_complex));
        }
    }

    return nitems;
}


} /* namespace kernel */
} /* namespace filter */
} /* namespace gr */
