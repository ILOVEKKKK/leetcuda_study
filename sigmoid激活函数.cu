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
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

//手动限定FP32格式下的最大值与最小值
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

//普通sigmoid算子，无优化
__global__ void sigmoid_f32_kernel(float* x, float* y,int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < N)
    {
        float v = x[idx];
        v = fminf(fmaxf(v,MIN_EXP_F32),MAX,_EXP_F32);
        y[idx] = 1.0f/(1.0f+expf(-v));
    }
}

//sigmoid算子，一次加载4个FP32数据优化加载
__global__ void sigmoid_f32x4_kernel(float* x,float* y,int N)
{
    int idx = 4*(blockIdx.x*blockDim.x+threadIdx.x);
    //边界检查
    if (idx + 3 < N)
    {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;

        reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX, _EXP_F32);
        reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX, _EXP_F32);
        reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX, _EXP_F32);
        reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX, _EXP_F32);

        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
        FLOAT4(y[idx]) = reg_y;
    }
    else
    {
        if(idx < N) y[idx] = reg_y.x;
        if(idx + 1 < N) y[idx+1] = reg_y.y;
        if(idx + 2 < N) y[idx+2] = reg_y.z;
    }
}

//sigmoid算子FP16版本
__global__ void sigmoid_f16_kernel(float* x,float* y,int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const half f = __float2half(1.0f);
    if(idx < N)
    {
        half v = x[idx];
        v = __hmin(__hmax(v,MIN_EXP_F16),MAX_EXP_F16);
        y[idx] = f/(f+hexp(-v));
    }
}

//sigmoid算子fp16版本，向量化加载一次读取两个
__global__ void sigmoid_f16x2_kernel(half* x,half* y,int N)
{
    int idx = 2 * (blockIdx.x*blockDim.x+threadIdx.x);
    if(idx + 1 < N)
    {
        const half v = __float2half(1.0f);
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_x.x = __hmin(__hmax(reg_x.x,MIN_EXP_F16),MAX_EXP_F16);
        reg_x.y = __hmin(__hmax(reg_x.y,MIN_EXP_F16),MAX_EXP_F16);
        reg_y.x = v/v+hexp(-reg_x.x);
        reg_y.y = v/v+hexp(-reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
    else
    {
        if(idx < N)
        {
            y[idx] = reg_y.x;
        }
    }
}

//sigmoid算子，一次读取128字节
__global__ void sigmoid_f16x8_kernel(half* x,half* y,int N)
{
    int idx = 8*(blockIdx.x*blockDim.x+threadIdx.x);
    const half f = __float2half(1.0f);
    //边界检查
    if(idx + 7 < N)
    {
        half pack_x[8],pack_y[8];//利用寄存器存储读取进来的数据
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
        #pragma unroll//展开循环体用GPU并行特性节省循环开销
        for(int i = 0;i<8;++i)
        {
            half v = __hmin(__hmax(pack_x[i],MIN_EXP_F16),MAX_EXP_F16);
            pack_y[i] = f/(f+hexp(-v));
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
    else
    {
        for(int i = idx;i < N;++i)
        {
            half v = __hmin(__hmax(x[i],MIN_EXP_F16),MAX_EXP_F16);
            y[i] = f/(f+hexp(-v));
        }
    }
}





