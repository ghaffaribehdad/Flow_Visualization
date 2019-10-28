#pragma once

#include "cuda_runtime.h"
#include <corecrt_math.h>
#include <ctgmath>

#define X_HAT {1.0f,0.0f,0.0f}
#define Y_HAT {0.0f,1.0f,0.0f}
#define Z_HAT {0.0f, 0.0f, 1.0f}


//float3 operations

inline __host__ __device__ bool outofTexture(float3 position)
{
	if (position.x > 1 || position.x < 0)
		return true;
	if (position.y > 1 || position.y < 0)
		return true;
	if (position.z > 1 || position.z < 0)
		return true;

	return false;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float2 operator*(float2 a, int2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ bool operator==(float3 a, float3 b)
{
	if (a.x != b.x)
		return false;
	if (a.y != b.y)
		return false;
	if (a.z != b.z)
		return false;

	return true;
}


inline __host__ __device__ bool operator==(int3 a, int3 b)
{
	if (a.x != b.x)
		return false;
	if (a.y != b.y)
		return false;
	if (a.z != b.z)
		return false;

	return true;
}

inline __host__ __device__ float3 operator*(float3 a, int3 b)
{
	return make_float3
	(
		a.x * b.x,
		a.y * b.y,
		a.z * b.z
	);
}



inline __host__ __device__ float3 operator/(float3 a, int3 b)
{
	return make_float3
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z
	);
}

inline __host__ __device__ float2 operator/(float2 a, int2 b)
{
	return make_float2
	(
		a.x / b.x,
		a.y / b.y
	);
}

// multiply float3 and int3 and then round it up  e.g. in one dimension ceilMult(2.3,3) = 7
inline __host__ __device__ int3 ceilMult(float3 a, int3 b)
{
	return make_int3
	(
		static_cast<int>(ceil(a.x * b.x)),
		static_cast<int>(ceil(a.y * b.y)),
		static_cast<int>(ceil(a.z * b.z))
	);
}

inline __host__ __device__ int3 floorMult(float3 a, int3 b)
{
	return make_int3
	(
		static_cast<int>(floor(a.x * b.x)),
		static_cast<int>(floor(a.y * b.y)),
		static_cast<int>(floor(a.z * b.z))
	);
}


inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}


inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float4 operator*(const float4& a ,const float & b)
{
	return make_float4(a.x * b, a.y * b, a.z * b,a.w * b);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}


inline __host__ __device__ float2 operator*(float a, float2 b)
{
	return make_float2(a * b.x, a * b.y);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}



inline __host__ __device__ void operator+=(float3& a, const float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __host__ __device__ float3 operator/(const float3& a, const float& b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float2 operator/(const float2& a, const float2& b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}


inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(const float4&  a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}




inline __host__ __device__ float3 normalize(const float3 & v)
{
	float invLen = 1.0f/sqrtf(dot(v, v));
	return v * invLen;
}


inline __host__ __device__ float2 normalize(const float2& v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float VecMagnitude(const float3& v)
{
	return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ void operator-=( float3& a, const float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

inline __host__ __device__ void operator-=(float4& a, const float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float DecodeFloatRGBA(float4 rgba) {
	return dot(rgba, make_float4(1.0f, 1.0f / 255.0f, 1.0f / 65025.0f, 1.0f / 16581375.0f));
}



