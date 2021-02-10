/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_IMPL_H
#define INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_IMPL_H

#include <legacy_cudaops/fft_filter_ccf.h>
#include <legacy_cudaops/fft_filter.h>

namespace gr {
namespace legacy_cudaops {

class fft_filter_ccf_impl : public fft_filter_ccf
{
private:
    int d_nsamples;
    bool d_updated;
    std::vector<float> d_new_taps;
    kernel::fft_filter_ccf d_filter;

public:
    fft_filter_ccf_impl(int decimation, const std::vector<float>& taps);
    ~fft_filter_ccf_impl();
    void set_taps(const std::vector<float>& taps);
    std::vector<float> taps() const;

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;

};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_IMPL_H */
