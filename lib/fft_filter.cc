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

#include <gnuradio/logger.h>
#include <legacy_cudaops/fft_filter.h>
#include <volk/volk.h>
#include <cstring>
#include <iostream>
#include <memory>

#include "helper_cuda.h"
extern void exec_add_kernel_cc(cuFloatComplex* in,
                               cuFloatComplex* out,
                               cuFloatComplex* a,
                               int veclen,
                               int batch_size,
                               cudaStream_t stream);

void exec_multiply_kernel_ccc(cuFloatComplex* in,
                              cuFloatComplex* out,
                              cuFloatComplex* a,
                              int veclen,
                              int batch_size,
                              cudaStream_t stream);

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
    cudaStreamCreate(&stream);
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

    checkCudaErrors(cudaMemset(d_dev_tail, 0, d_fftsize));

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

    checkCudaErrors(cudaMemcpy(d_dev_taps,
                               &d_xformed_taps[0],
                               d_fftsize * sizeof(gr_complex),
                               cudaMemcpyHostToDevice));

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

        checkCudaErrors(cufftCreate(&d_plan));

        checkCudaErrors(cufftSetStream(d_plan, stream));

        size_t workSize;
        checkCudaErrors(cufftMakePlanMany(d_plan,
                                          1,
                                          &d_fftsize,
                                          NULL,
                                          1,
                                          1,
                                          NULL,
                                          1,
                                          1,
                                          CUFFT_C2C,
                                          d_batch_size,
                                          &workSize));

        checkCudaErrors(
            cudaMalloc((void**)&d_data, sizeof(cufftComplex) * d_fftsize * d_batch_size));
        checkCudaErrors(cudaMalloc((void**)&d_dev_tail,
                                   sizeof(cufftComplex) * d_fftsize * d_batch_size));

        d_xformed_taps.resize(d_fftsize);

        checkCudaErrors(cudaMalloc((void**)&d_dev_taps,
                                   sizeof(cufftComplex) * d_fftsize * d_batch_size));
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
        checkCudaErrors(cudaMemcpyAsync(d_data,
                                        &input[i],
                                        d_nsamples * sizeof(gr_complex),
                                        cudaMemcpyHostToDevice,
                                        stream));

        checkCudaErrors(cudaMemsetAsync(d_data + d_nsamples,
                                        0,
                                        sizeof(gr_complex) * (d_fftsize - d_nsamples),
                                        stream));

        checkCudaErrors(cufftExecC2C(d_plan, d_data, d_data, CUFFT_FORWARD));

        exec_multiply_kernel_ccc(d_data, d_data, d_dev_taps, d_fftsize, 1, stream);

        checkCudaErrors(cufftExecC2C(d_plan, d_data, d_data, CUFFT_INVERSE));

        exec_add_kernel_cc(d_data, d_data, d_dev_tail, tailsize(), 1, stream);

        j = dec_ctr;
        checkCudaErrors(cudaMemcpyAsync(&output[i],
                                        d_data,
                                        d_nsamples * sizeof(gr_complex),
                                        cudaMemcpyDeviceToHost,
                                        stream));
        dec_ctr = (j - d_nsamples);

        // // stash the tail
        checkCudaErrors(cudaMemcpyAsync(d_dev_tail,
                                        d_data + d_nsamples,
                                        tailsize() * sizeof(gr_complex),
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }
    cudaStreamSynchronize(stream);

    return nitems;
}


} /* namespace kernel */
} // namespace legacy_cudaops
} /* namespace gr */
