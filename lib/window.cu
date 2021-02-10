#include <cuComplex.h>

__global__ void
window_kernel_ccf(cuFloatComplex* in, cuFloatComplex* out, float* window, int veclen, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = veclen * batch_size;

    // int which_batch = i / veclen;
    int batch_idx = i % veclen;

    if (i < n) {
        out[i].x = in[i].x * window[batch_idx];
        out[i].y = in[i].y * window[batch_idx];
    }
}

void exec_window_kernel_ccf(cuFloatComplex* in, cuFloatComplex* out, float* window, int veclen, int batch_size)
{
    int block_size = 1024; // max num of threads
    int nblocks = (veclen * batch_size + block_size - 1) / block_size;
    window_kernel_ccf<<<nblocks, block_size>>>(in, out, window, veclen, batch_size);
}