#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

//普通relu算子，无优化
__global__ void relu_fp32_kernel(float* x,float* y,int N)
{
    int idx = blockIdx.x*blockDimx.x+threadIdx.x;
    if(idx < N)
    {
        y[idx] = fmaf(0.0f,x[idx]);
    }
}

//relu算子，一次加载4个fp32优化
__global__ void relu_fp32x4_kernel(float* x,float* y,int N)
{
    int idx = 4*(blockIdx.x*blockDim.x+threadIdx.x);
    if(idx + 3 < N)
    {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = fmaxf(reg_x.x,0.0f);
        reg_y.y = fmaxf(reg_x.y,0.0f);
        reg_y.z = fmaxf(reg_x.z,0.0f);
        reg_y.w = fmaxf(reg_x.w,0.0f);
        FLOAT4(y[idx]) = reg_y;
    }
    else
    {
        for(int i = idx;i < N;++i)
        {
            y[i] = fmaxf(0.0f,x[i]);
        }
    }
}

//relu算子，fp16无优化
__global__ void relu_f16_kernel(half *x, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = __hmax(__float2half(0.0f), x[idx]);
}

//relu算子，利用half2一次读取两个fp16优化
__global__ void relu_f16x2_kernel(half *x, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 1 < N) {
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y = HALF2(y[idx]);
    reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
    reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
    HALF2(y[idx]) = reg_y;
  }
  else
  {
    for(int i = idx;i < N;++i)
    {
        y[i] = fmaxf(0.0f,x[i]);
    }
  }
}

//relu算子，一次读取128字节进行优化
__global__ void relu_fp16x8_kernel(half* x,half* y,int N)
{
    int idx = 8*(blockIdx.x*blockDim.x+threadIdx.x);
    const HALF2 zero_2 = {__float2half(0.0f), __float2half(0.0f)};//用来relu比较的两个常数0，half2格式
    if(idx + 7 < N)
    {
        half pack_x[8],pack_y[8];
        LDST128BITS(pack_x[8],x[idx]);
        for(int i = 0;i<8;i+=2)
        {
            pack_y[i] = __hmax(HALF2(pack_x[i]),zero_2);
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}