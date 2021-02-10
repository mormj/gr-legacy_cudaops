/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_filter_ccf_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace legacy_cudaops {

using input_type = gr_complex;
using output_type = gr_complex;
fft_filter_ccf::sptr fft_filter_ccf::make(int decimation, const std::vector<float>& taps)
{
    return gnuradio::make_block_sptr<fft_filter_ccf_impl>(decimation, taps);
}


/*
 * The private constructor
 */
fft_filter_ccf_impl::fft_filter_ccf_impl(int decimation, const std::vector<float>& taps)
    : gr::sync_decimator("fft_filter_ccf",
                         gr::io_signature::make(1, 1, sizeof(input_type)),
                         gr::io_signature::make(1, 1, sizeof(output_type)),
                         decimation),
      d_updated(false),
      d_filter(decimation, taps)
{
    d_new_taps = taps;
    d_nsamples = d_filter.set_taps(taps);
    set_output_multiple(d_nsamples);
}

/*
 * Our virtual destructor.
 */
fft_filter_ccf_impl::~fft_filter_ccf_impl() {}

int fft_filter_ccf_impl::work(int noutput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    const gr_complex* in = (const gr_complex*)input_items[0];
    gr_complex* out = (gr_complex*)output_items[0];

    if (d_updated) {
        d_nsamples = d_filter.set_taps(d_new_taps);
        d_updated = false;
        set_output_multiple(d_nsamples);
        return 0; // output multiple may have changed
    }

    d_filter.filter(noutput_items, in, out);

    return noutput_items;
}


void fft_filter_ccf_impl::set_taps(const std::vector<float>& taps)
{
    d_new_taps = taps;
    d_updated = true;
}

std::vector<float> fft_filter_ccf_impl::taps() const { return d_new_taps; }


} /* namespace legacy_cudaops */
} /* namespace gr */
