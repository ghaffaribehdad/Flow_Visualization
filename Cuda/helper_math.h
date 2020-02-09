#pragma once

#include "cuda_runtime.h"
#include <corecrt_math.h>
#include <ctgmath>
#include <DirectXMath.h>

#define X_HAT {1.0f,0.0f,0.0f}
#define Y_HAT {0.0f,1.0f,0.0f}
#define Z_HAT {0.0f, 0.0f, 1.0f}

#define CUDA_INDEX (blockIdx.x* blockDim.y* blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)


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

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
	return make_float4
	(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z,
		a.w / b.w
		);
}

inline __host__ __device__ float4 operator/(float4 a, float b)
{
	return make_float4
	(
		a.x / b,
		a.y / b,
		a.z / b,
		a.w / b
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

inline __host__ __device__ float4 operator+(const float4& a, const float& b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
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



inline __host__ __device__ bool operator<(const float3& a, const float3& b)
{
	if (a.x >= b.x)
		return false;
	if (a.y >= b.y)
		return false;
	if (a.z >= b.z)
		return false;

	return true;
}

inline __host__ __device__ bool operator>(const float3& a, const float3& b)
{
	if (a.x <= b.x)
		return false;
	if (a.y <= b.y)
		return false;
	if (a.z <= b.z)
		return false;

	return true;
}



inline __device__ __host__ int2 IndexToPixel(int& index, int2& dim)
{
	int2 pixel;
	pixel.y = index / dim.x;
	pixel.x = index - pixel.y * dim.x;

	return pixel;
}


 __device__ inline float4 ValueAtXY_Surface_float4(cudaSurfaceObject_t surf, int2  gridPos)
{	
	float4 data;

	surf2Dread(&data, surf, gridPos.x * sizeof(float4), gridPos.y);

	return data;
};

inline __device__ float4 ValueAtXYZ_Texture_float4(cudaTextureObject_t tex, int3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z);
}



inline __device__  float2 Gradient2DX_Surf(cudaSurfaceObject_t surf, int2 gridPosition, int2 gridSize)
{
	float dH_dX = 0.0f;
	float dH_dY = 0.0f;


	if (gridPosition.x != 0 && gridPosition.x != gridSize.x - 1)
	{
		dH_dX = ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x + 1, gridPosition.y)).x;
		dH_dX -= ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x - 1, gridPosition.y)).x;
	}
	else if (gridPosition.x == 0)
	{
		dH_dX = ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x + 1, gridPosition.y)).x;
		dH_dX -= ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x, gridPosition.y)).x;
		dH_dX = 2 * dH_dX;
		
	}
	else if (gridPosition.x == gridSize.x - 1)
	{
		dH_dX =		ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x, gridPosition.y)).x;
		dH_dX -=	ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x - 1, gridPosition.y)).x;
		dH_dX = 2 * dH_dX;
	}



	// Y direction
	if (gridPosition.y != 0 && gridPosition.y != gridSize.y - 1)
	{
		dH_dY = ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x, gridPosition.y + 1)).x;
		dH_dY -= ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x, gridPosition.y - 1)).x;
	}
	else if (gridPosition.y == 0)
	{
		dH_dY = ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x, gridPosition.y + 1)).x;
		dH_dY -= ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x,  gridPosition.y)).x;
		dH_dY = 2 * dH_dY;
		
	}
	else if (gridPosition.y == gridSize.y - 1)
	{
		dH_dY = ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x,  gridPosition.y)).x;
		dH_dY -= ValueAtXY_Surface_float4(surf, make_int2(gridPosition.x,  gridPosition.y - 1)).x;
		dH_dY = 2 * dH_dY;
	}


	return make_float2(dH_dX, dH_dY);
}



inline __device__  float2 GradientXY_Tex3D_X(cudaSurfaceObject_t tex, int3 gridPosition, int2 gridSize)
{
	float dH_dX = 0.0f;
	float dH_dY = 0.0f;


	if (gridPosition.x != 0 && gridPosition.x != gridSize.x - 1)
	{
		dH_dX = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).x;
		dH_dX -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).x;
	}
	else if (gridPosition.x == 0)
	{
		dH_dX = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).x;
		dH_dX -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).x;
		dH_dX = 2 * dH_dX;

	}
	else if (gridPosition.x == gridSize.x - 1)
	{
		dH_dX = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).x;
		dH_dX -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).x;
		dH_dX = 2 * dH_dX;
	}



	// Y direction
	if (gridPosition.y != 0 && gridPosition.y != gridSize.y - 1)
	{
		dH_dY = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).x;
		dH_dY -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).x;
	}
	else if (gridPosition.y == 0)
	{
		dH_dY = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).x;
		dH_dY -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).x;
		dH_dY = 2 * dH_dY;

	}
	else if (gridPosition.y == gridSize.y - 1)
	{
		dH_dY = ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).x;
		dH_dY -= ValueAtXYZ_Texture_float4(tex, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).x;
		dH_dY = 2 * dH_dY;
	}


	return make_float2(dH_dX, dH_dY);
}


