#pragma once

#define ARRAYTOFLOAT3(X) {X[0],X[1],X[2]}
#define ARRAYTOINT3(X) {X[0],X[1],X[2]}
#define I_3X3_F fMat3X3(1.0f,0,0,0,1.0f,0,0,0,1.0f)
#define I_3X3_D dMat3X3(1.0,0,0,0,1.0,0,0,0,1.0)
#define safeAcos(x) acos(fmax(-1.0, fmin(1.0, (x))))
#define safeSqrt(x) sqrt(fmax(0.0, (x)))
#define maxBlockDim (uint)16

#include "cuda_runtime.h"
#include <corecrt_math.h>
#include <ctgmath>
#include <DirectXMath.h>



#define X_HAT {1.0f,0.0f,0.0f}
#define Y_HAT {0.0f,1.0f,0.0f}
#define Z_HAT {0.0f, 0.0f, 1.0f}
#define CUDA_PI_F 3.141592654f
#define CUDA_PI_D 3.141592654


#define CUDA_INDEX (blockIdx.x* blockDim.y* blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)

#define BLOCK_THREAD(kernelCall) static_cast<unsigned int>((kernelCall % (thread.x* thread.y) == 0 ? kernelCall / (thread.x * thread.y) : kernelCall / (thread.x * thread.y) + 1))

#define  EPS2 1e-2
#define  EPS1 1e-5

inline __host__ __device__ bool isZero(float x)
{
	if (x == 0)
		return true;

	return false;
}

inline __host__ __device__ float fsign(float v)
{
	return copysignf(1.0f, v);
}


static __host__ __device__ __inline__ double sign(float v) { return ::fsign(v); }


// convert array to built-in cuda structures
 inline __host__ __device__ float3 Array2Float3(float* a_float)
{
	return make_float3(a_float[0], a_float[1], a_float[2]);
}

 inline __host__ __device__ float2 FLOAT3XY(const float3& a)
 {
	 return make_float2(a.x, a.y);
 }

 inline __host__ __device__ int2 INT3XY(const int3& a)
 {
	 return make_int2(a.x, a.y);
 }

inline __host__ __device__ int3 Array2Int3(int* a_int)
{
	return make_int3(a_int[0], a_int[1], a_int[2]);
}

inline __host__ __device__ float4 Array2Float4(float* a)
{
	return make_float4(a[0], a[1], a[2], a[3]);
}

inline __host__ __device__ int2 ARRAYTOINT2(int* a_int)
{
	return make_int2(a_int[0], a_int[1]);
}

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

__device__ __host__ inline double3 operator*(double& a, double3 & b)
{
	return make_double3
	(
		a * b.x,
		a * b.y,
		a * b.z
	);
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}



inline __host__ __device__ float4 operator*(float a, float4 b)
{
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
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
		a.x * (float)b.x,
		a.y * (float)b.y,
		a.z * (float)b.z
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

inline __host__ __device__ float4 operator+=(float4 a, float4 b)
{
	return make_float4
	(
		a.x += b.x,
		a.y += b.y,
		a.z += b.z,
		a.w += b.w
	);
}


inline __host__ __device__ float4 operator-=(float4 a, float4 b)
{
	return make_float4
	(
		a.x -= b.x,
		a.y -= b.y,
		a.z -= b.z,
		a.w -= b.w
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
		a.x / (float)b.x,
		a.y / (float)b.y,
		a.z / (float)b.z
	);
}

inline __host__ __device__ float2 operator/(float2 a, int2 b)
{
	return make_float2
	(
		a.x / (float)b.x,
		a.y / (float)b.y
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

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
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

inline __host__ __device__ float2 operator*(float a, int2 b)
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

inline __host__ __device__ double dot(const double3& a, const double3& b)
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

inline __host__ __device__ float magnitude(const float3& v)
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


inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
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

inline __device__ float4 ValueAtXYZ_Texture_float4(cudaTextureObject_t tex, float3 position)
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


struct fMat3X3
{
	float3 r1;
	float3 r2;
	float3 r3;


	__host__ __device__ fMat3X3
	(
		const float3& _r1,
		const float3& _r2,
		const float3& _r3
	) :r1(_r1),r2(_r2),r3(_r3) {}

	__host__ __device__ fMat3X3
	(
		const float& _r1x,
		const float& _r1y,
		const float& _r1z,
		const float& _r2x,
		const float& _r2y,
		const float& _r2z,
		const float& _r3x,
		const float& _r3y,
		const float& _r3z
	) : r1{ _r1x,_r1y,_r1z }, r2{ _r2x,_r2y,_r2z }, r3{ _r3x,_r3y,_r3z } {}

	__host__ __device__ const float3 c1()
	{
		return make_float3(r1.x, r2.x, r3.x);
	}

	__host__ __device__ float3 c2()
	{
		return make_float3(r1.y, r2.y, r3.y);
	}

	__host__ __device__ float3 c3()
	{
		return make_float3(r1.z, r2.z, r3.z);
	}

	__host__ __device__ float det()
	{
		float d1 = r1.x * ((r2.y * r3.z) - ( r3.y * r2.z));
		float d2 = r1.y * ((r2.x * r3.z) - ( r3.x * r2.z));
		float d3 = r1.z * ((r2.x * r3.y) - ( r3.x * r2.y));

		return d1 - d2 + d3;
	}
};



__device__ __host__ inline fMat3X3 transpose(fMat3X3& a)
{ 
	return fMat3X3
	(
		make_float3(a.r1.x, a.r2.x, a.r3.x),
		make_float3(a.r1.y, a.r2.y, a.r3.y),
		make_float3(a.r1.z, a.r2.z, a.r3.z)
	);
}

__device__ __host__ inline fMat3X3 operator*(float & a, fMat3X3& b)
{
	return fMat3X3
	(
		a* b.r1,
		a* b.r2,
		a* b.r3
	);
}






__device__ __host__ inline fMat3X3 operator*(fMat3X3& b, float & a)
{
	return fMat3X3
	(
		a * b.r1,
		a * b.r2,
		a * b.r3
	);
}



__device__ __host__ inline fMat3X3 operator-(const fMat3X3& a, const fMat3X3& b)
{
	return fMat3X3
	(
		a.r1 - b.r1,
		a.r2 - b.r2,
		a.r3 - b.r3
	);
}

__device__ __host__ inline fMat3X3 operator+(const fMat3X3& a, const fMat3X3& b)
{
	return fMat3X3
	(
		a.r1 + b.r1,
		a.r2 + b.r2,
		a.r3 + b.r3
	);
}


__device__ __host__ inline fMat3X3 mult(fMat3X3& a, fMat3X3& b)
{
	return fMat3X3
	(
		dot(a.r1, b.c1()), dot(a.r1, b.c2()), dot(a.r1, b.c3()),
		dot(a.r2, b.c1()), dot(a.r2, b.c2()), dot(a.r2, b.c3()),
		dot(a.r3, b.c1()), dot(a.r3, b.c2()), dot(a.r3, b.c3())
	);
}





struct dMat3X3
{
	double3 r1;
	double3 r2;
	double3 r3;


	__host__ __device__ dMat3X3
	(
		const double3 & _r1,
		const double3 & _r2,
		const double3 & _r3
	) :r1(_r1), r2(_r2), r3(_r3) {}

	__host__ __device__ dMat3X3
	(
		const double& _r1x,
		const double& _r1y,
		const double& _r1z,
		const double& _r2x,
		const double& _r2y,
		const double& _r2z,
		const double& _r3x,
		const double& _r3y,
		const double& _r3z
	) : r1{ _r1x,_r1y,_r1z }, r2{ _r2x,_r2y,_r2z }, r3{ _r3x,_r3y,_r3z } {}

	__host__ __device__ const double3 c1()
	{
		return make_double3(r1.x, r2.x, r3.x);
	}

	__host__ __device__ double3 c2()
	{
		return make_double3(r1.y, r2.y, r3.y);
	}

	__host__ __device__ double3 c3()
	{
		return make_double3(r1.z, r2.z, r3.z);
	}

	__host__ __device__ double det()
	{
		double d1 = r1.x * ((r2.y * r3.z) - (r3.y * r2.z));
		double d2 = r1.y * ((r2.x * r3.z) - (r3.x * r2.z));
		double d3 = r1.z * ((r2.x * r3.y) - (r3.x * r2.y));

		return d1 - d2 + d3;
	}



};


inline __host__ __device__ dMat3X3 transpose(dMat3X3 & mat)
{

	double3 r1t = { mat.r1.x,mat.r2.x,mat.r3.x };
	double3 r2t = { mat.r1.y,mat.r2.y,mat.r3.y };
	double3 r3t = { mat.r1.z,mat.r2.z,mat.r3.z };

	return dMat3X3(r1t, r2t, r3t);

}


__device__ __host__ inline dMat3X3 operator*(double & a, dMat3X3& b)
{
	return dMat3X3(
		a* b.r1,
		a* b.r2,
		a* b.r3
	);
}





__device__ __host__ inline dMat3X3 operator*(dMat3X3& b, double & a)
{
	return dMat3X3
	(
		a * b.r1,
		a * b.r2,
		a * b.r3
	);
}



__device__ __host__ inline dMat3X3 operator-(const dMat3X3& a, const dMat3X3& b)
{
	return dMat3X3
	(
		a.r1 - b.r1,
		a.r2 - b.r2,
		a.r3 - b.r3
	);
}



__device__ __host__ inline dMat3X3 mult(dMat3X3& a, dMat3X3& b)
{
	return dMat3X3
	(
		dot(a.r1, b.c1()), dot(a.r1, b.c2()), dot(a.r1, b.c3()),
		dot(a.r2, b.c1()), dot(a.r2, b.c2()), dot(a.r2, b.c3()),
		dot(a.r3, b.c1()), dot(a.r3, b.c2()), dot(a.r3, b.c3())
	);
}










// Min to Max
__device__ __host__ inline void sort3Items(float3& v)
{
	float t;
	if (v.y < v.x)
	{
		t = v.x;
		if (v.z < v.y) { v.x = v.z; v.z = t; }
		else
		{
			if (v.z < t) { v.x = v.y; v.y = v.z; v.z = t; }
			else { v.x = v.y; v.y = t; }
		}
	}
	else
	{
		if (v.z < v.y)
		{
			t = v.z;
			v.z = v.y;

			if (t < v.x) { v.y = v.x; v.x = t; }
			else { v.y = t; }
		}
	}
}


/***********************************************************************************************
* Eigensolver by Hasan et al.
* additional sorting of the eigenvalues (no positive definite tensor)
***********************************************************************************************/

__device__ __host__ inline void eigensolveHasan(const fMat3X3& J, float3& sortedEigenvalues)
{
	const float3 vOne = make_float3(1, 1, 1);
	float3 vDiag = make_float3(J.r1.x, J.r2.y, J.r3.z);  // xx , yy , zz
	float3 vOffDiag = make_float3(J.r1.y, J.r1.z, J.r2.z);  // xy , xz , yz
	float3 offSq = vOffDiag * vOffDiag;
	float I1 = dot(vDiag, vOne);
	float I2 = dot(make_float3(vDiag.x, vDiag.x, vDiag.y), make_float3(vDiag.y, vDiag.z, vDiag.z)) - dot(offSq, vOne);
	float I3 = vDiag.x * vDiag.y * vDiag.z + 2.0f * vOffDiag.x * vOffDiag.y * vOffDiag.z - dot(make_float3(vDiag.z, vDiag.y, vDiag.x), offSq);
	float I1_3 = I1 / 3.0f;
	float I1_3Sq = I1_3 * I1_3;
	float v = I1_3Sq - I2 / 3.0f;
	float vInv = 1.0f / v;
	float s = I1_3Sq * I1_3 - I1 * I2 / 6.0f + I3 / 2.0f;
	float phi = acosf(s * vInv * sqrt(vInv)) / 3.0f;
	float vSqrt2 = 2.0f * sqrt(v);

	sortedEigenvalues = make_float3(I1_3 + vSqrt2 * cosf(phi), I1_3 - vSqrt2 * cosf((CUDA_PI_F / 3.0f) + phi), I1_3 - vSqrt2 * cosf((CUDA_PI_F / 3.0f) - phi));
	sort3Items(sortedEigenvalues);

}



__device__ __host__ inline float eigenValueMax(fMat3X3 & J)
{
	float3 eig = { 0.0f, 0.0f,0.0f };

	float p1 = pow(J.r1.y, 2.0f) + pow(J.r1.z,2.0f) + pow(J.r2.z, 2.0f);
	
	if (p1 == 0)
	{
		eig.x = J.r1.x;
		eig.y = J.r2.y;
		eig.z = J.r3.z;
	}
	else
	{
		float q = (J.r1.x + J.r2.y + J.r3.z) / 3.0f;
		float p2 = pow(J.r1.x - q, 2.0f) + pow(J.r2.y - q, 2.0f) + pow(J.r3.z - q, 2.0f) + 2 * p1;
		float p = sqrtf(p2 / 6.0f);
		fMat3X3 I = I_3X3_F;
		fMat3X3 B = J - (q * I);// I is the identity matrix	
		float p_ = (1.0f / p);
		B = p_ * B;
		
		float r = B.det() * 0.5f;
		float phi = 0.0f;

		if (r <= -1)
			phi = CUDA_PI_F / 3.0f;
		else if (r >= 1)
			phi = 0;
		else
			phi = acos(r) / 3.0f;

		// the eigenvalues satisfy eig3 <= eig2 <= eig1
		eig.x = q + 2.0f * p * cos(phi);
		eig.z = q + 2.0f * p * cos(phi + (2.0f * CUDA_PI_F / 3.0f));
		eig.y = 3.0f * q - eig.x - eig.z; // % since trace(A) = eig1 + eig2 + eig3;

	}

	return eig.x;
}


__device__ __host__ inline float eigenValue2(fMat3X3 & J)
{
	float3 eig = { 0.0f, 0.0f,0.0f };

	float p1 = pow(J.r1.y, 2.0f) + pow(J.r1.z, 2.0f) + pow(J.r2.z, 2.0f);

	if (p1 == 0)
	{
		eig.x = J.r1.x;
		eig.y = J.r2.y;
		eig.z = J.r3.z;
	}
	else
	{
		float q = (J.r1.x + J.r2.y + J.r3.z) / 3.0f;
		float p2 = pow(J.r1.x - q, 2.0f) + pow(J.r2.y - q, 2.0f) + pow(J.r3.z - q, 2.0f) + 2 * p1;
		float p = sqrtf(p2 / 6.0f);
		fMat3X3 I = I_3X3_F;
		fMat3X3 B = J - (q * I);// I is the identity matrix	
		float p_ = (1.0f / p);
		B = p_ * B;

		float r = B.det() * 0.5f;
		float phi = 0.0f;

		if (r <= -1)
			phi = CUDA_PI_F / 3.0f;
		else if (r >= 1)
			phi = 0;
		else
			phi = acos(r) / 3.0f;

		// the eigenvalues satisfy eig3 <= eig2 <= eig1
		eig.x = q + 2.0f * p * cos(phi);
		eig.z = q + 2.0f * p * cos(phi + (2.0f * CUDA_PI_F / 3.0f));
		eig.y = 3.0f * q - eig.x - eig.z; // % since trace(A) = eig1 + eig2 + eig3;

	}

	return eig.y;
}


__device__ __host__ inline double eigenValueMax(dMat3X3 & J)
{
	double3 eig = { 0.0, 0.0,0.0 };

	double p1 = pow(J.r1.y, 2.0) + pow(J.r1.z, 2.0) + pow(J.r2.z, 2.0);

	if (p1 == 0)
	{
		eig.x = J.r1.x;
		eig.y = J.r2.y;
		eig.z = J.r3.z;
	}
	else
	{
		double q = (J.r1.x + J.r2.y + J.r3.z) / 3.0f;
		double p2 = pow(J.r1.x - q, 2.0) + pow(J.r2.y - q, 2.0) + pow(J.r3.z - q, 2.0) + 2 * p1;
		double p = sqrt(p2 / 6.0);
		dMat3X3 I = I_3X3_D;
		dMat3X3 B = J - (q * I);// I is the identity matrix	
		double p_ = (1.0f / p);
		B = p_ * B;

		double r = B.det() * 0.5;
		double phi = 0.0;

		if (r <= -1)
			phi = CUDA_PI_F / 3.0;
		else if (r >= 1)
			phi = 0;
		else
			phi = acos(r) / 3.0;

		// the eigenvalues satisfy eig3 <= eig2 <= eig1
		eig.x = q + 2.0 * p * cos(phi);
		eig.z = q + 2.0 * p * cos(phi + (2.0 * CUDA_PI_D / 3.0));
		eig.y = 3.0 * q - eig.x - eig.z; // % since trace(A) = eig1 + eig2 + eig3;

	}

	return eig.x;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}


inline __host__ __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
	return make_float3(floor(v.x), floor(v.y), floor(v.z));
}


inline __device__  float3 world2Tex(float3 position, float3 dimension, const int3 & size, bool noormalized = false, bool particleTracing = false)
{

	if (noormalized)
	{
		return (position / dimension);
	}
	float3 pos = (position / dimension) * make_int3(size.x - 1, size.y - 1, size.z - 1) + make_float3(0.5f, 0.5f, 0.5f);


	if (particleTracing)
	{
		if (pos.x > 0)
		{
			pos.x = fmod(pos.x, (float)(size.x - 1));
		}
		else
		{
			pos.x = (float)(size.x - 1) + fmod(pos.x, (float)(size.x - 1));
		}

		if (pos.y > 0)
		{
			pos.y = fmod(pos.y, (float)(size.y - 1));
		}
		else
		{
			pos.y = (float)(size.y - 1) + fmod(pos.y, (float)(size.y - 1));
		}


		if (pos.z > 0)
		{
			pos.z = fmod(pos.z, (float)(size.z - 1));
		}
		else
		{
			pos.z = (float)(size.z - 1) + fmod(pos.z, (float)(size.z - 1));
		}

	}
	return pos;
}


__device__ __host__ inline float3 saturateRGB(const float3 & rgb, const float & saturate)
{
	float3 rgb_complement = make_float3(1, 1, 1) - rgb;
	return ((1 - saturate) * rgb_complement) + rgb;
}

__device__ __host__ inline float3 float4tofloat3(const float4 & a)
{
	return make_float3(a.x, a.y, a.z);
}

__device__ inline fMat3X3 Jacobian(cudaTextureObject_t t_velocityField, float3 h, float3 position)
{
	fMat3X3 jac = { 0,0,0,0,0,0,0,0,0 };

	jac.r1 = float4tofloat3(tex3D<float4>(t_velocityField, position.x + 1, position.y, position.z));
	jac.r1 -= float4tofloat3(tex3D<float4>(t_velocityField, position.x - 1, position.y, position.z));

	jac.r2 = float4tofloat3(tex3D<float4>(t_velocityField, position.x, position.y + 1, position.z));
	jac.r2 -= float4tofloat3(tex3D<float4>(t_velocityField, position.x, position.y - 1, position.z));

	jac.r3 = float4tofloat3(tex3D<float4>(t_velocityField, position.x, position.y, position.z + 1));
	jac.r3 -= float4tofloat3(tex3D<float4>(t_velocityField, position.x, position.y, position.z - 1));

	// This would give us the Jacobian Matrix
	jac.r1 = jac.r1 / (2 * h.x);
	jac.r2 = jac.r2 / (2 * h.y);
	jac.r2 = jac.r3 / (2 * h.z);

	return jac;
}


inline __host__ __device__ float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f / 3.0f - 0.5f*t*t*a;
	else if (t < 2.0f) return a * a*a / 6.0f;
	else return 0.0f;
}


//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 64 nearest neighbour lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate

inline __device__ float4 cubicTex3DSimple(cudaTextureObject_t tex, float3 coord)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float4 result = { 0.0f,0.0f,0.0f,0.0f };
	for (float z = -1; z < 2.5f; z++)  //range [-1, 2]
	{
		float bsplineZ = bspline(z - fraction.z);
		float w = index.z + z;
		for (float y = -1; y < 2.5f; y++)
		{
			float bsplineYZ = bspline(y - fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x = -1; x < 2.5f; x++)
			{
				float bsplineXYZ = bspline(x - fraction.x) * bsplineYZ;
				float u = index.x + x;
		
				result = result + bsplineXYZ * tex3D<float4>(tex, u, v, w);

			}
		}
	}
	return result;
}


inline __device__ dMat3X3 jacobian(cudaTextureObject_t t_VelocityField, const float3& relativePos, const float3& h)
{
	dMat3X3 jac = { 0,0,0,0,0,0,0,0,0 };
	
	float4 dVx = tex3D<float4>(t_VelocityField, relativePos.x + h.x / 2.0, relativePos.y, relativePos.z);
	 dVx -= tex3D<float4>(t_VelocityField, relativePos.x - h.x / 2.0, relativePos.y, relativePos.z);

	float4 dVy = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y + h.x / 2.0, relativePos.z);
	 dVy -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y - h.x / 2.0, relativePos.z);

	float4 dVz = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z + h.x / 2.0);
	 dVz -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z - h.x / 2.0);

	// This would give us the Jacobian Matrix
	dVx = dVx / h.x;
	dVy = dVy / h.y;
	dVz = dVz / h.z;

	jac =
	{
		dVx.x,dVx.y,dVx.z,
		dVy.x,dVy.y,dVy.z,
		dVz.x,dVz.y,dVz.z,
	};



	return jac;
}



inline __device__ fMat3X3 fjacobian(cudaTextureObject_t t_VelocityField, const float3& relativePos, const float3& h)
{
	fMat3X3 jac = { 0,0,0,0,0,0,0,0,0 };

	float4 dVx = tex3D<float4>(t_VelocityField, relativePos.x + 1, relativePos.y, relativePos.z);
	dVx -= tex3D<float4>(t_VelocityField, relativePos.x - 1, relativePos.y, relativePos.z);

	float4 dVy = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y + 1, relativePos.z);
	dVy -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y - 1, relativePos.z);

	float4 dVz = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z + 1);
	dVz -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z - 1);

	// This would give us the Jacobian Matrix
	dVx = 0.5 * dVx / h.x;
	dVy = 0.5 * dVy / h.y;
	dVz = 0.5 * dVz / h.z;


	jac =
	{
		dVx.x,dVx.y,dVx.z,
		dVy.x,dVy.y,dVy.z,
		dVz.x,dVz.y,dVz.z,
	};

	return jac;
}




inline __device__  float lambda2(cudaTextureObject_t tex, const float3 & gridDiamter, const int3 & gridSize, const float3 & position)
{
	fMat3X3 f_s = fMat3X3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);	// S Matrix
	fMat3X3 f_omega = fMat3X3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);	// S Matrix
	fMat3X3 f_Jacobian(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);	// Jacobian Matrix

	float3 h = gridDiamter / gridSize;
	f_Jacobian = fjacobian(tex, position, h);

	f_s =  f_Jacobian + transpose(f_Jacobian);
	f_omega = f_Jacobian - transpose(f_Jacobian);
	
	float quarter = 0.25f;
	f_s =		mult(f_s, f_s) ;
	f_omega =	mult(f_omega, f_omega);
	
	f_s = quarter * f_s;
	f_omega = quarter * f_omega;
	// calculate the sum
	f_s = f_omega + f_s;

	return -eigenValue2(f_s);



}



/**
 * Returns the cubic polynomial factors for the ray r(t)=entry+t*dir
 * traversing the voxel with corner values given by 'vals' and accessing the
 * tri-linear interpolated values.
 */

/**
		 * \param vals the eight corner values xyz=[(0,0,0), (1,0,0), (0,1,0), ..., (1,1,1)]
		 * \param entry the entry point to the voxel in [0,voxelSize]^3
		 * \param dir the direction within the voxel
 */

template <typename T>
inline __host__ __device__ float4 getFactors(
	const float vals[8], const float3& entry, const float3& dir)
{
	const T v0 = vals[0], v1 = vals[1], v2 = vals[2], v3 = vals[3];
	const T v4 = vals[4], v5 = vals[5], v6 = vals[6], v7 = vals[7];
	const T ex = entry.x, ey = entry.y, ez = entry.z;
	const T dx = dir.x, dy = dir.y, dz = dir.z;

#define GET_FACTOR_VERSION 3
#if GET_FACTOR_VERSION==1
	// expanded version
	const T a = -dx * dy * dz * v0 + dx * dy * dz * v1 + dx * dy * dz * v2 - dx * dy * dz * v3 + dx * dy * dz * v4 - dx * dy * dz * v5 - dx * dy * dz * v6 + dx * dy * dz * v7;
	const T b = -dx * dy * ez * v0 + dx * dy * ez * v1 + dx * dy * ez * v2 - dx * dy * ez * v3 + dx * dy * ez * v4 - dx * dy * ez * v5 - dx * dy * ez * v6 + dx * dy * ez * v7 + dx * dy * v0 - dx * dy * v1 - dx * dy * v2 + dx * dy * v3 - dx * dz * ey * v0 + dx * dz * ey * v1 + dx * dz * ey * v2 - dx * dz * ey * v3 + dx * dz * ey * v4 - dx * dz * ey * v5 - dx * dz * ey * v6 + dx * dz * ey * v7 + dx * dz * v0 - dx * dz * v1 - dx * dz * v4 + dx * dz * v5 - dy * dz * ex * v0 + dy * dz * ex * v1 + dy * dz * ex * v2 - dy * dz * ex * v3 + dy * dz * ex * v4 - dy * dz * ex * v5 - dy * dz * ex * v6 + dy * dz * ex * v7 + dy * dz * v0 - dy * dz * v2 - dy * dz * v4 + dy * dz * v6;
	const T c = -dx * ey * ez * v0 + dx * ey * ez * v1 + dx * ey * ez * v2 - dx * ey * ez * v3 + dx * ey * ez * v4 - dx * ey * ez * v5 - dx * ey * ez * v6 + dx * ey * ez * v7 + dx * ey * v0 - dx * ey * v1 - dx * ey * v2 + dx * ey * v3 + dx * ez * v0 - dx * ez * v1 - dx * ez * v4 + dx * ez * v5 - dx * v0 + dx * v1 - dy * ex * ez * v0 + dy * ex * ez * v1 + dy * ex * ez * v2 - dy * ex * ez * v3 + dy * ex * ez * v4 - dy * ex * ez * v5 - dy * ex * ez * v6 + dy * ex * ez * v7 + dy * ex * v0 - dy * ex * v1 - dy * ex * v2 + dy * ex * v3 + dy * ez * v0 - dy * ez * v2 - dy * ez * v4 + dy * ez * v6 - dy * v0 + dy * v2 - dz * ex * ey * v0 + dz * ex * ey * v1 + dz * ex * ey * v2 - dz * ex * ey * v3 + dz * ex * ey * v4 - dz * ex * ey * v5 - dz * ex * ey * v6 + dz * ex * ey * v7 + dz * ex * v0 - dz * ex * v1 - dz * ex * v4 + dz * ex * v5 + dz * ey * v0 - dz * ey * v2 - dz * ey * v4 + dz * ey * v6 - dz * v0 + dz * v4;
	const T d = -ex * ey * ez * v0 + ex * ey * ez * v1 + ex * ey * ez * v2 - ex * ey * ez * v3 + ex * ey * ez * v4 - ex * ey * ez * v5 - ex * ey * ez * v6 + ex * ey * ez * v7 + ex * ey * v0 - ex * ey * v1 - ex * ey * v2 + ex * ey * v3 + ex * ez * v0 - ex * ez * v1 - ex * ez * v4 + ex * ez * v5 - ex * v0 + ex * v1 + ey * ez * v0 - ey * ez * v2 - ey * ez * v4 + ey * ez * v6 - ey * v0 + ey * v2 - ez * v0 + ez * v4 + v0;
#elif GET_FACTOR_VERSION==2
	// factored version
	// a bit faster, but more numerically unstable !?!
	const T t1 = -v0 + v1 + v2 - v3 + v4 - v5 - v6 + v7;
	const T t2 = v0 - v1 - v2 + v3;
	const T t3 = v0 - v1 - v4 + v5;
	const T t4 = v0 - v2 - v4 + v6;
	const T a = (dx * dy * dz) * t1;
	const T b = (dx * dy * ez) * t1 + (dx * dy) * t2 + (dx * dz * ey) * t1 + (dx * dz) * t3 + (dy * dz * ex) * t1 + (dy * dz) * t4;
	const T c = (dx * ey * ez) * t1 + (dx * ey) * t2 + (dx * ez) * t3 + dx * (-v0 + v1) + (dy * ex * ez) * t1 + (dy * ex) * t2 + (dy * ez) * t4 + dy * (-v0 + v2) + (dz * ex * ey) * t1 + (dz * ex) * t3 + (dz * ey) * t4 + dz * (-v0 + v4);
	const T d = (ex * ey * ez) * t1 + (ex * ey) * t2 + (ex * ez) * t3 + ex * (-v0 + v1) + (ey * ez) * t4 + ey * (-v0 + v2) + ez * (-v0 + v4) + v0;
#elif GET_FACTOR_VERSION==3
	//Based on "Interactive ray tracing for isosurface rendering"
	//by Steven Parker et al., 1998

	//reorder values, z first
	const T values[8] = { v0, v4, v2, v6, v1, v5, v3, v7 };
	//assemble basis functions
	const T uA[2] = { 1 - ex, ex };
	const T vA[2] = { 1 - ey, ey };
	const T wA[2] = { 1 - ez, ez };
	const T uB[2] = { -dx, dx };
	const T vB[2] = { -dy, dy };
	const T wB[2] = { -dz, dz };
	//compute factors
	T a = 0;
	T b = 0;
	T c = 0;
	T d = 0; // -isovalue;
	int valueIndex = 0;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				a += uB[i] * vB[j] * wB[k] * values[valueIndex];
				b += (uA[i] * vB[j] * wB[k] + uB[i] * vA[j] * wB[k] + uB[i] * vB[j] * wA[k]) * values[valueIndex];
				c += (uB[i] * vA[j] * wA[k] + uA[i] * vB[j] * wA[k] + uA[i] * vA[j] * wB[k]) * values[valueIndex];
				d += uA[i] * vA[j] * wA[k] * values[valueIndex];
				valueIndex++;
			}
		}
	}
#endif
	return make_float4( a, b, c, d );
};

// returns the number of roots
template <typename T>
inline __host__ __device__ int rootsHyperbolic(const float4& factors, T roots[3])
{
	//extract factors
	const T a = factors.x, b = factors.y, c = factors.z, d = factors.w;

	if (fabs(a) <= EPS2)
	{
		if (isZero(b))
		{
			//linear equation
			if (isZero(c)) return 0; //constant
			roots[0] = -d / c;
			return 1;
		}
		//quadratic equation
		T discr = c * c - T(4) * b * d;
		if (discr < 0) return 0;
		if (isZero(discr))
		{
			roots[0] = -c / (T(2) * b);
			return 1;
		}
		else {
			discr = sqrt(discr);
			//https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very
			roots[0] = (-c - sign(c) * discr) / (2 * b);
			roots[1] = d / (b * roots[0]);
			//roots[0] = (-c + discr) / (T(2) * b);
			//roots[1] = (-c - discr) / (T(2) * b);
			return 2;
		}
	}

	//convert to depressed cubic t^3+pt+q=0
	const T p = (T(3) * a * c - b * b) / (T(3) * a * a);
	const T q = (T(2) * b * b * b - T(9) * a * b * c + T(27) * a * a * d) / (T(27) * a * a * a);

#define t2x(t) ((t)-b/(3*a))

	if (fabs(p) <= EPS1)
	{
		//there exists exactly one root
		roots[0] = t2x(cbrt(-q));
		return 1;
	}
	//formular of Francois Vite
	//https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_solution_for_three_real_roots
	const T Delta = T(4) * p * p * p + T(27) * q * q;
	if (Delta > 0)
	{
		//one real root
		T t0;
		if (p < 0)
			t0 = T(-2) * sign(q) * sqrt(-p / T(3)) * cosh(T(1) / T(3) * acosh(T(-3) * abs(q) / (T(2) * p) * sqrt(T(-3) / p)));
		else
			t0 = T(-2) * sqrt(p / T(3)) * sinh(T(1) / T(3) * asinh(T(3) * q / (T(2) * p) * sqrt(T(3) / p)));
		roots[0] = t2x(t0);
		return 1;
	}
	//TODO: handle double root if Delta>-EPS1:
	// simple root at 3q/p
	// double root at -3a/2p
	else
	{
		//three real roots
		const T f1 = T(2) * safeSqrt(-p / T(3));
		const T f2 = T(1) / T(3) * safeAcos(T(3) * q / (T(2) * p) * safeSqrt(-T(3) / p));
		for (int k = 0; k < 3; ++k)
			roots[k] = t2x(f1 * cos(f2 - T(2) * T(CUDA_PI_F) * k / T(3)));
		return 3;
	}

#undef t2x
};




__device__  inline float3 colorCode(const float* minColor, const float*maxColor, const float & value, const float & min_val, const float& max_val)
{
	float3 rgb_min =
	{
		minColor[0],
		minColor[1],
		minColor[2],
	};

	float3 rgb_max =
	{
		maxColor[0],
		maxColor[1],
		maxColor[2],
	};

	float3 rgb = { 0,0,0 };
	float y_saturated = 0.0f;

	if (value < 0)
	{
		float3 rgb_min_complement = make_float3(1, 1, 1) - rgb_min;
		y_saturated = saturate(abs(value / min_val));
		rgb = rgb_min_complement * (1 - y_saturated) + rgb_min;
	}
	else
	{
		float3 rgb_max_complement = make_float3(1, 1, 1) - rgb_max;
		y_saturated = saturate(value / max_val);
		rgb = rgb_max_complement * (1 - y_saturated) + rgb_max;
	}

	return rgb;
}



