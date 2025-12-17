#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
//a为输入的N元数组,y为记录统计结果的直方图数组


//标准INT32直方图统计
__global__ void histogram_i32_kernel(int *a,int *y,int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < N)
    {
        atomicAdd(&y[a[idx]],1);
    }
}
