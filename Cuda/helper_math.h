#pragma once

#define ARRAYTOFLOAT3(X) {X[0],X[1],X[2]}
#define ARRAYTOINT3(X) {X[0],X[1],X[2]}
#define I_3X3_F fMat3X3(1.0f,0,0,0,1.0f,0,0,0,1.0f)
#define I_3X3_D dMat3X3(1.0,0,0,0,1.0,0,0,0,1.0)

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


__device__ __host__ inline dMat3X3 operator*(double & a, dMat3X3& b)
{
	return dMat3X3(
		a* b.r1,
		a* b.r2,
		a* b.r3
	);
}

__device__ __host__ inline dMat3X3 transpose(dMat3X3& a)
{
	return dMat3X3
	(
		make_double3(a.r1.x, a.r2.x, a.r3.x),
		make_double3(a.r1.y, a.r2.y, a.r3.y),
		make_double3(a.r1.z, a.r2.z, a.r3.z)
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


__device__ __host__ inline float3 world2Tex(const float3& position, const float3& dimension, const int3& size, bool noormalized = false)
{


	if (noormalized)
	{
		return (position / dimension);
	}

	return (position / dimension) *size;
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

	return jac;
}

