#pragma once

#include "cuda_runtime_api.h"
#include "BoundingBox.h"
#include "..//Cuda/helper_math.h"
#include <DirectXMath.h>



__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox);

__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j);

__device__ uchar4 rgbaFloatToUChar(float4 rgba);


inline float3 XMFloat3ToFloat3(const DirectX::XMFLOAT3& src)
{
	return make_float3(src.x, src.y, src.z);
}

inline float3 ArrayFloat3ToFloat3(float* src)
{
	return make_float3(src[0], src[1], src[2]);
}

inline int3 ArrayInt3ToInt3(int* src)
{
	return make_int3(src[0], src[1], src[2]);
}