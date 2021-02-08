/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LEGACY_CUDAOPS_CUFFT_IMPL_H
#define INCLUDED_LEGACY_CUDAOPS_CUFFT_IMPL_H

#include <legacy_cudaops/cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace legacy_cudaops {

class cufft_impl : public cufft
{

private:
    size_t d_fft_size;
    bool d_forward;
    bool d_shift;
    size_t d_batch_size;
    memory_model_t d_mem_model;

    cufftHandle d_plan;
    cufftComplex *d_data;   


public:
    cufft_impl(const size_t fft_size,
                     const bool forward,
                     bool shift = false,
                     const size_t batch_size = 1,
                     const memory_model_t mem_model = memory_model_t::TRADITIONAL);
    ~cufft_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);

};

} // namespace legacy_cudaops
} // namespace gr

#endif /* INCLUDED_LEGACY_CUDAOPS_CUFFT_IMPL_H */
