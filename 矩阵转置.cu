#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define TILE_DIM 32
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

//矩阵转置算子，fp32数据格式，读取X[row][col]写入y[col][row]
__global__ void mat_transpose_fp32_col2row_kernel(float* x,float* y,const int row,const int col)
{
    int ix = blockIdx.x*blockDim.x+threadIdx.x;
    int iy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = iy*col+ix;//读取时是连续读取的
    if(idx < row*col)
    {
        y[ix*row+iy] = x[idx]; //涉及到跨步访问，需要优化
    }
}

//矩阵转置算子，fp32数据格式，使用共享内存优化内存访问模式
__global__ void mat_transpose_fp32_col2row_kernel(float* x,float* y,const int row,const int col)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    //映射到输入矩阵的二维坐标
    int ix = blockIdx.x*TILE_DIM+threadIdx.x;
    int iy = blockIdx.y*TILE_DIM+threadIdx.y;
    int idx = iy*col+ix;
    if(iy < row && ix < col)
    {
        tile[threadIdx.y][threadIdx.x] = x[idx];//连续内存访问
    }
    __synthreads();
    //映射到输出矩阵的二维坐标，只转置了block，block里面的线程负责的值还没有被转置
    int ix2 = blockIdx.y*blockDim.y+threadIdx.x;
    int iy2 = blockIdx.x*blockDim.x+threadIdx.y;
    int idx2 = iy2*row+ix2;//依然按行优先顺序遍历存放转置后矩阵的内存位置
    if(ix2 < row && iy2 < col)
    {
        y[idx2] = tile[threadIdx.x][threadIdx.y];
    }

}

__global__ void mat_transpose_fp32_col2row_kernel(float* x,float* y,const int row,const int col)
{
    
}