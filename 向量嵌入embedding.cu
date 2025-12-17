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
//这段代码实现了一个在 GPU 上并行执行的查表操作。
//它的核心功能是：根据给定的索引，从一个大的“权重”表weight中，批量地、高效地、并行地复制出相应的行向量。

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

//向量嵌入算子,fp32格式无优化
__global__ void embedding_fp32_kernel(const int *idx,float* weight,float* output,int n,int embed_size)
{
    //idx数组存储的是要查找的向量在weight数组中的哪一行，即索引
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int offset = idx[bx]*embed_size;
    output[bx*embed_size+tx] = weight[offset+tx];
}

//向量嵌入算子，fp32一次加载4个优化
__global__ void embedding_fp32x4_kernel(const int *idx,float* weight,float* output,int n,int embed_size)
{
    int tx = threadIdx.x*4;
    int bx = blockIdx.x;
    if (bx < N)
    {
        int offset = idx[bx] * embed_size;
        if (tx + 3 < embed_size)
        {
            FLOAT4(output[bx * embed_size + tx]) = FLOAT4(weight[offset + tx]);
        }
        else
        {
            for(int i = tx;i < embed_size;++i)
            {
                output[bx*embed_size+i] = weight[offset+i];
            }
        }
    }
}

__global__ void 