#pragma once
#include <cuda_runtime.h>

__device__ float callerValueAtTex(int i, cudaTextureObject_t tex, float3 position, float3  gridDiameter = { 0,0,0 }, int3 gridSize = { 0,0,0 });
__device__ float3 callerGradientAtTex(int i, cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
