/* -*- c++ -*- */
/*
 * Copyright 2021 gr-legacy_cudaops author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "copy_impl.h"
#include <gnuradio/io_signature.h>


extern void apply_copy(cuFloatComplex* in,
                       cuFloatComplex* out,
                       int grid_size,
                       int block_size,
                       int load,
                       cudaStream_t stream);
extern void get_block_and_grid(int* minGrid, int* minBlock);


namespace gr {
namespace legacy_cudaops {

using input_type = gr_complex;
using output_type = gr_complex;
copy::sptr copy::make(int batch_size, int load, memory_model_t mem_model)
{
    return gnuradio::make_block_sptr<copy_impl>(batch_size, load, mem_model);
}


/*
 * The private constructor
 */
copy_impl::copy_impl(int batch_size, int load, memory_model_t mem_model)
    : gr::sync_block(
          "copy",
          gr::io_signature::make(1, 1 /* min, max nr of inputs */, sizeof(input_type)),
          gr::io_signature::make(1, 1 /* min, max nr of outputs */, sizeof(output_type))),
      d_batch_size(batch_size),
      d_mem_model(mem_model),
      d_load(load)
{
    // Allocate Device Memory
    if (d_mem_model == memory_model_t::TRADITIONAL) {

        checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(gr_complex) * d_batch_size));
    } else if (d_mem_model == memory_model_t::PINNED) {
        checkCudaErrors(
            cudaMallocHost((void**)&d_data, sizeof(gr_complex) * d_batch_size));
    } else {
        checkCudaErrors(
            cudaMallocManaged((void**)&d_data, sizeof(gr_complex) * d_batch_size));
    }

    get_block_and_grid(&d_min_grid_size, &d_block_size);
    cudaDeviceSynchronize();
    std::cerr << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
              << std::endl;

    // if (batch_size < d_block_size)
    // {
    //     throw std::runtime_error("batch_size must be a multiple of block size");
    // }
    cudaStreamCreate(&stream);
    set_output_multiple(d_batch_size);
}

/*
 * Our virtual destructor.
 */
copy_impl::~copy_impl() {}

int copy_impl::work(int noutput_items,
                    gr_vector_const_void_star& input_items,
                    gr_vector_void_star& output_items)
{
    const input_type* in = reinterpret_cast<const input_type*>(input_items[0]);
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);

    auto nvecs = noutput_items / d_batch_size;
    auto mem_size = d_batch_size * sizeof(gr_complex); // in bytes, for the memcpy

    for (auto s = 0; s < nvecs; s++) {

        if (d_mem_model == memory_model_t::TRADITIONAL) {
            checkCudaErrors(cudaMemcpyAsync(
                d_data, in + s * d_batch_size, mem_size, cudaMemcpyHostToDevice, stream));

            apply_copy(d_data, d_data, d_batch_size / d_block_size, d_block_size, d_load, stream);
        } else // if (d_mem_model == memory_model_t::PINNED)
        {
            memcpy(d_data, in + s * d_batch_size, mem_size);
            apply_copy(d_data, d_data, d_batch_size / d_block_size, d_block_size, d_load, stream);
        }
        // else
        // {

        //     apply_copy((cuFloatComplex *)(in+s* d_batch_size), (cuFloatComplex
        //     *)(out+s* d_batch_size), d_batch_size / d_block_size, d_block_size);
        // }

        // cudaDeviceSynchronize();


        if (d_mem_model == memory_model_t::TRADITIONAL) {
            checkCudaErrors(cudaMemcpyAsync(out + s * d_batch_size,
                                            d_data,
                                            mem_size,
                                            cudaMemcpyDeviceToHost,
                                            stream));
        } else // if (d_mem_model == memory_model_t::PINNED)
        {
            memcpy(out + s * d_batch_size, d_data, mem_size);
        }

        cudaStreamSynchronize(stream);
    }
    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace legacy_cudaops */
} /* namespace gr */
