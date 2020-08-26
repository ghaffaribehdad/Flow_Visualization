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

inline void multArrayFloat3Int3(float*z,float * x, int* y)
{
	z[0] = x[0] * y[0];
	z[0] = x[1] * y[1];
	z[0] = x[2] * y[2];
}
inline void divideArrayFloat3Int3(float*z, float * x, int* y)
{
	z[0] = x[0] / y[0];
	z[0] = x[1] / y[1];
	z[0] = x[2] / y[2];
}