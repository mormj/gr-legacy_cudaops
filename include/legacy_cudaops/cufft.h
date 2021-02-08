/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_CUFFT_H
#define INCLUDED_LEGACY_CUDAOPS_CUFFT_H

#include <gnuradio/sync_block.h>
#include <legacy_cudaops/api.h>
#include <legacy_cudaops/memmodel.h>

namespace gr {
namespace legacy_cudaops {

/*!
 * \brief <+description of block+>
 * \ingroup legacy_cudaops
 *
 */
class LEGACY_CUDAOPS_API cufft : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<cufft> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of legacy_cudaops::cufft.
     *
     * To avoid accidental use of raw pointers, legacy_cudaops::cufft's
     * constructor is in a private implementation
     * class. legacy_cudaops::cufft::make is the public interface for
     * creating new instances.
     */
    static sptr make(const size_t fft_size,
                     const bool forward,
                     bool shift = false,
                     const size_t batch_size = 1,
                     const memory_model_t mem_model = memory_model_t::TRADITIONAL);
};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_CUFFT_H */
