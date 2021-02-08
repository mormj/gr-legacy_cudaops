/* -*- c++ -*- */
/*
 * Copyright 2021 gr-legacy_cudaops author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_COPY_IMPL_H
#define INCLUDED_LEGACY_CUDAOPS_COPY_IMPL_H

#include <cuComplex.h>
#include <legacy_cudaops/copy.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

namespace gr {
namespace legacy_cudaops {

class copy_impl : public copy
{
private:
    int d_batch_size;
    int d_min_grid_size;
    int d_block_size;
    int d_load;
    memory_model_t d_mem_model;
    cuFloatComplex* d_data;
    cudaStream_t stream;

public:
    copy_impl(int batch_size, int load, memory_model_t mem_model);
    ~copy_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_COPY_IMPL_H */
