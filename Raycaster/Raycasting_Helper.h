#pragma once

#include "cuda_runtime_api.h"
#include "BoundingBox.h"
#include <DirectXMath.h>
 


__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox);
__device__ float2 findEnterExit(const float3 & pixelPos, const float3  & dir, float boxFaces[6]);
__device__ float findExitPoint(const float2 & entery, const float2 & dir, const float2 & cellSize);
__device__ float findExitPoint3D(const float3 & entery, const float3 & dir, const float3 & cellSize);

__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j);

__device__ uchar4 rgbaFloatToUChar(float4 rgba);

