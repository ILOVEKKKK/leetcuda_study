#include <iostream>
#include <vector>

__global__ void elementwise_add_f32_kernel(const float *a, const float *b, float *c,
                                           int N) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}