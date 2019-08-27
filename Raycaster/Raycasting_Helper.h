#pragma once

#include "cuda_runtime_api.h"
#include "BoundingBox.h"
#include "..//Cuda/helper_math.h"



__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox);

__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j);

__device__ uchar4 rgbaFloatToUChar(float4 rgba);
