#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define WARP_SIZE 32
/*CUDA提供了一些内置的向量类型，如 float4, int4, half2 等。当编译器看到你对这些类型的变量进行读写时，它会自动生成高效的、宽位数的加载/存储指令（例如，LD.128 代表加载128位）。
这些宏的作用就是一种“戏法”，欺骗编译器，让它将普通的指针（如float*）当作指向向量类型的指针（如float4*），从而触发这种高效的内存操作。*/
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

//fp32加法-无优化
__global__ void elementwise_add_f32_kernel(const float *a, const float *b, float *c,
                                           int N) 
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}

//fp32加法-向量化加载优化
__global__ void elementwise_add_f32x4_pack_kernel(float *a, float *b, float *c,
                                                  int N)
{
    int idx = 4*(blockIdx.x*blockDim.x+threadIdx.x);
    //边界检查，确认够4个元素进行向量化加载
    if(idx + 3 < N)
    {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.w = reg_a.w+reg_b.w;
        reg_c.x = reg_a.x+reg_b.x;
        reg_c.y = reg_a.y+reg_b.y;
        reg_c.z = reg_a.z+reg_b.z;
        FLOAT4(c[idx]) = reg_c;
    }
    else
    {
      for(int i = idx;i<N;++i)
      {
        c[i] = a[i] + b[i];
      }
    }
}

__global__ void elementwise_add_f16x2_pack_kernel(float *a, float *b, float *c,
                                           int N)
{
    int idx = 2*(blockIdx.x*blockDim.x+threadIdx.x);
    if(idx + 1 < N)
    {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        /*使用 __hadd 可以确保编译器生成的是原生的、最高效的FP16加法指令。它避免了编译器可能做出的“自作主张”的优化，比如为了精度，它可能会先把两个half提升(promote)到32位的float，
        进行32位加法，然后再把结果转换(demote)回16位的half。这个“提升-计算-降级”的过程会引入不必要的开销，比直接执行16位加法要慢。 */
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(c[idx]) = reg_c;
    }
    else
    {
      for(int i = idx;i<N;++i)
      {
        c[i] = a[i] + b[i];
      }
    }
}

//128位总线一次读取8个fp16，并分成4个half2来进行处理
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_a_0 = HALF2(a[idx + 0]);
  half2 reg_a_1 = HALF2(a[idx + 2]);
  half2 reg_a_2 = HALF2(a[idx + 4]);
  half2 reg_a_3 = HALF2(a[idx + 6]);
  half2 reg_b_0 = HALF2(b[idx + 0]);
  half2 reg_b_1 = HALF2(b[idx + 2]);
  half2 reg_b_2 = HALF2(b[idx + 4]);
  half2 reg_b_3 = HALF2(b[idx + 6]);
  half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
  reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
  reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
  reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
  reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
  reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
  reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
  reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
  reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
  if ((idx + 0) < N) {
    HALF2(c[idx + 0]) = reg_c_0;
  }
  if ((idx + 2) < N) {
    HALF2(c[idx + 2]) = reg_c_1;
  }
  if ((idx + 4) < N) {
    HALF2(c[idx + 4]) = reg_c_2;
  }
  if ((idx + 6) < N) {
    HALF2(c[idx + 6]) = reg_c_3;
  }
}

//128位总线一次读取128位数据，即8个fp16数据，并进行处理
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N)
{
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_a[8], pack_b[8], pack_c[8];
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
  //两个两个fp16为一个单位加载到pack_c中，并进行运算
  for (int i = 0; i < 8; i+=2)
  {
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]),HALF2(pack_b[i]));
  }
  //一次将128位数据写入c中
  if(idx + 7 < N)
  {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  }
  else
  {
    for(int i = 0;idx + i < N;i++)
    {
      c[idx+i] = __hadd(a[idx+i],b[idx+i]);
    }
  }
}