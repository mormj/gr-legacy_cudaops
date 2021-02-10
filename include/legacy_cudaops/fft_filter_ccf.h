/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_H
#define INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_H

#include <gnuradio/sync_decimator.h>
#include <legacy_cudaops/api.h>

namespace gr {
namespace legacy_cudaops {

/*!
 * \brief Fast FFT filter with gr_complex input, gr_complex output and float taps
 * \ingroup filter_blk
 *
 * \details
 * This block implements a complex decimating filter using the
 * fast convolution method via an FFT. The decimation factor is an
 * integer that is greater than or equal to 1.
 *
 * The filter takes a set of complex (or real) taps to use in the
 * filtering operation. These taps can be defined as anything that
 * satisfies the user's filtering needs. For standard filters such
 * as lowpass, highpass, bandpass, etc., the filter.firdes and
 * filter.optfir classes provide convenient generating methods.
 *
 * This filter is implemented by using the FFTW package to perform
 * the required FFTs. An optional argument, nthreads, may be
 * passed to the constructor (or set using the set_nthreads member
 * function) to split the FFT among N number of threads. This can
 * improve performance on very large FFTs (that is, if the number
 * of taps used is very large) if you have enough threads/cores to
 * support it.
 */

class LEGACY_CUDAOPS_API fft_filter_ccf : virtual public gr::sync_decimator
{
public:
    typedef std::shared_ptr<fft_filter_ccf> sptr;

    /*!
     * Build an FFT filter blocks.
     *
     * \param decimation  >= 1
     * \param taps        complex filter taps
     */
    static sptr make(int decimation, const std::vector<float>& taps);
    virtual void set_taps(const std::vector<float>& taps) = 0;
    virtual std::vector<float> taps() const = 0;
};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_CCF_H */
