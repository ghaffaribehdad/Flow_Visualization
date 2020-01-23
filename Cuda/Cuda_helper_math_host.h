#pragma once
#include <cuda_runtime.h>
#include <DirectXMath.h>


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

inline int3 ArrayInt2ToInt3(int* src)
{
	return make_int3(src[0], src[1], 0);
}