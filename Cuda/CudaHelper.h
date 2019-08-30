#pragma once
#include "cuda_runtime.h"
#include "helper_math.h"
#include <math.h>


// RKOdd assumes the first timevolume is in first texture and second timevolume in second
__device__ float3 RK4Odd(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, float3* position, float3 gridDiameter, float dt);

// RKEven assumes the first timevolume is in first texture and second timevolume in second
__device__ float3 RK4Even(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, float3 * position, float3 gridDiameter, float dt);

__device__ float3 RK4EStream(cudaTextureObject_t t_VelocityField_0, float3* position, float3 gridDiameter, float dt);

