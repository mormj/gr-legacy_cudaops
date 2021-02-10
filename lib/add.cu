#include <cuComplex.h>

__global__ void
add_kernel_cc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = veclen * batch_size;

    // int which_batch = i / veclen;
    int batch_idx = i % veclen;

    if (i < n) {
        float re = in[i].x + a[batch_idx].x;
        float im = in[i].y + a[batch_idx].y;
        out[i].x = re;
        out[i].y = im;
    }
}

void exec_add_kernel_cc(cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex* a, int veclen, int batch_size)
{
    int block_size = 1024; // max num of threads
    int nblocks = (veclen * batch_size + block_size - 1) / block_size;
    add_kernel_cc<<<nblocks, block_size>>>(in, out, a, veclen, batch_size);
}