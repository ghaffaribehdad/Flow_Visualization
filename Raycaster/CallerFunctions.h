#pragma once
#include <cuda_runtime.h>

__device__ float callerValueAtTex(int measure, cudaTextureObject_t tex, float3 position, float3  gridDiameter = { 0,0,0 }, int3 gridSize = { 0,0,0 });
__device__ float callerValueAtTex(int measure, cudaTextureObject_t tex, float3 position, float3  gridDiameter , int3 gridSize , float sigma, int3 offset0, int3 offset1 );
__device__ float callerValueAtTex(int measure, cudaTextureObject_t tex0, cudaTextureObject_t tex1, float3 position, float3 gridDiameter = { 0,0,0 }, int3 gridSize = { 0,0,0 });

__device__ float3 callerGradientAtTex(int measure, cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
__device__ float3 callerGradientAtTex(int measure, cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize, const float & sigma, const int3 & offset0, const int3 & offset1);
__device__ float3 callerGradientAtTex(int measure, cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
