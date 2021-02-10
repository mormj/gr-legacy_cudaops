/* -*- c++ -*- */
/*
 * Copyright 2010,2012,2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_H
#define INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_H

#include <gnuradio/fft/fft.h>
#include <gnuradio/filter/api.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/logger.h>
#include <volk/volk_alloc.hh>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace legacy_cudaops {
namespace kernel {

/*!
 * \brief Fast FFT filter with gr_complex input, gr_complex output and float taps
 * \ingroup filter_blk
 *
 * \details
 * This block performs fast convolution using the
 * overlap-and-save algorithm. The filtering is performand in
 * the frequency domain instead of the time domain (see
 * gr::filter::kernel::fir_filter_ccf). For an input signal x
 * and filter coefficients (taps) t, we compute y as:
 *
 * \code
 *    y = ifft(fft(x)*fft(t))
 * \endcode
 *
 * This kernel computes the FFT of the taps when they are set to
 * only perform this operation once. The FFT of the input signal
 * x is done every time.
 *
 * Because this is designed as a very low-level kernel
 * operation, it is designed for speed and avoids certain checks
 * in the filter() function itself. The filter function expects
 * that the input signal is a multiple of d_nsamples in the
 * class that's computed internally to be as fast as
 * possible. The function set_taps will return the value of
 * nsamples that can be used externally to check this
 * boundary. Notice that all implementations of the fft_filter
 * GNU Radio blocks (e.g., gr::filter::fft_filter_ccf) use this
 * value of nsamples to compute the value to call
 * gr::block::set_output_multiple that ensures the scheduler
 * always passes this block the right number of samples.
 */
class fft_filter_ccf
{
private:
    int d_ntaps;
    int d_nsamples;
    int d_fftsize; // fftsize = ntaps + nsamples - 1
    int d_decimation;
    std::unique_ptr<fft::fft_complex_fwd> d_fwdfft; // forward "plan"
    std::unique_ptr<fft::fft_complex_rev> d_invfft; // inverse "plan"
    int d_nthreads = 1;                                 // number of FFTW threads to use
    std::vector<gr_complex> d_tail; // state carried between blocks for overlap-add
    std::vector<float> d_taps;      // stores time domain taps
    volk::vector<gr_complex> d_xformed_taps; // Fourier xformed taps

    void compute_sizes(int ntaps);
    int tailsize() const { return d_ntaps - 1; }

    gr::logger_ptr d_logger, d_debug_logger;

    size_t d_batch_size = 1;
    cufftHandle d_plan;
    cufftComplex *d_data;
    cufftComplex *d_dev_tail;
    cufftComplex *d_dev_taps;

public:
    /*!
     * \brief Construct an FFT filter for complex vectors with the given taps and
     * decimation rate.
     *
     * This is the basic implementation for performing FFT filter for fast convolution
     * in other blocks (e.g., gr::filter::fft_filter_ccf).
     *
     * \param decimation The decimation rate of the filter (int)
     * \param taps       The filter taps (float)
     * \param nthreads   The number of threads for the FFT to use (int)
     */
    fft_filter_ccf(int decimation, const std::vector<float>& taps, int nthreads = 1);

    // Disallow copy.
    //
    // This prevents accidentally doing needless copies, not just of fft_filter_xxx,
    // but every block that contains one.
    fft_filter_ccf(const fft_filter_ccf&) = delete;
    fft_filter_ccf& operator=(const fft_filter_ccf&) = delete;
    fft_filter_ccf(fft_filter_ccf&&) = default;
    fft_filter_ccf& operator=(fft_filter_ccf&&) = default;

    /*!
     * \brief Set new taps for the filter.
     *
     * Sets new taps and resets the class properties to handle different sizes
     * \param taps       The filter taps (complex)
     */
    int set_taps(const std::vector<float>& taps);

    /*!
     * \brief Returns the taps.
     */
    std::vector<float> taps() const;

    /*!
     * \brief Returns the number of taps in the filter.
     */
    unsigned int ntaps() const;

    /*!
     * \brief Returns the actual size of the filter.
     *
     * \details This value could be equal to ntaps, but we ofter
     * build a longer filter to allow us to calculate a more
     * efficient FFT. This value is the actual size of the filters
     * used in the calculation of the overlap-and-save operation.
     */
    unsigned int filtersize() const;

    /*!
     * \brief Perform the filter operation
     *
     * \param nitems  The number of items to produce
     * \param input   The input vector to be filtered
     * \param output  The result of the filter operation
     */
    int filter(int nitems, const gr_complex* input, gr_complex* output);
};

} /* namespace kernel */
} // namespace legacy_cudaops
} /* namespace gr */

#endif /* INCLUDED_LEGACY_CUDAOPS_FFT_FILTER_H */
