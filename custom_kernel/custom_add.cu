#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// C++ Wrapper
torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int size = a.numel();
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Launch Kernel
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size
    );
    
    return out;
}