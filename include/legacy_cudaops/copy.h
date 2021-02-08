/* -*- c++ -*- */
/*
 * Copyright 2021 gr-legacy_cudaops author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_COPY_H
#define INCLUDED_LEGACY_CUDAOPS_COPY_H

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
class LEGACY_CUDAOPS_API copy : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<copy> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of legacy_cudaops::copy.
     *
     * To avoid accidental use of raw pointers, legacy_cudaops::copy's
     * constructor is in a private implementation
     * class. legacy_cudaops::copy::make is the public interface for
     * creating new instances.
     */
    static sptr make(int batch_size, int load, memory_model_t mem_model);
};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_COPY_H */
