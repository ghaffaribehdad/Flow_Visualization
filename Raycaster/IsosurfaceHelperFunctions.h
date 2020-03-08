#pragma once
#include <cuda_runtime.h>

__device__  float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos);
__device__  float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position);
__device__ float3 GradientAtXYZ_X_Surface(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float3 GradientAtXYZ_Y_Surface(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float3 GradientAtXYZ_W_Surface(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float3 GradientAtXYZ_Z_Surface(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float3 GradientAtGrid_X(cudaTextureObject_t tex, float3 position, int3 gridSize);


namespace IsosurfaceHelper
{

	enum cudaSurfaceAddressMode
	{
		cudaAddressModeWrap = 0,    /**< Wrapping address mode */
		cudaAddressModeClamp = 1,    /**< Clamp to edge address mode */
		cudaAddressModeMirror = 2,    /**< Mirror address mode */
		cudaAddressModeBorder = 3     /**< Border address mode */
	};

	struct Observable
	{
		//Return the texel at XYZ of the Texture (Boundaries are controlled by the cudaTextureAddressMode
		__device__	virtual	float4		ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position);
		//Return the texel at XYZ of the Surface (Boundaries are controlled by the cudaTextureAddressMode
		__device__			float4		ValueAtXYZ_float4(cudaSurfaceObject_t surf, int3 position, cudaSurfaceAddressMode addressMode = cudaAddressModeBorder, bool interpolation = false);


		__device__ virtual	float		ValueAtXYZ(cudaTextureObject_t tex, float3 position)
		{
			return 0;
		};
		__device__ virtual	float		ValueAtXYZ_Surface(cudaSurfaceObject_t tex, float3 position)
		{
			return 0;
		};
		__device__ virtual	float3		GradientAtXYZ(cudaTextureObject_t tex, float3 position, int3 gridSize);
		__device__ virtual	float3		GradientAtGrid(cudaTextureObject_t tex, float3 relativePos, int3 gridSize);

	};

	struct Velocity_XYZT : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__  float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position) override;

	};

	struct Velocity_Magnitude : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__  float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;


	};

	struct Velocity_X : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__ float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;


	};

	struct Velocity_Y : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__ float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;

	};

	struct Velocity_Z : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__ float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;


	};

	struct ShearStress : public Observable
	{
		// calculates the value of the field at position XYZ
		__device__ float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;


	};

	struct TurbulentDiffusivity : public Observable
	{
		__device__ float ValueAtXYZ(cudaTextureObject_t tex, float3 position) override;
		__device__ float ValueAtXYZ_avgtemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
		__device__ float3 GradientAtGrid_AvgTemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
	};

	struct Position : public Observable
	{
		//calculates the value of the field at position XYZ
		__device__ float4 ValueAtXY(cudaTextureObject_t tex, float2 position);
		__device__ float ValueAtXY_Surface_float(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t tex, int3 gridPos);
		__device__ float4 ValueAtXY_Surface_float4(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float2 GradientAtXY_Grid(cudaSurfaceObject_t surf, int2 gridPosition);
		__device__ float2 GradientAtXYZ_Grid(cudaSurfaceObject_t surf, int3 gridPosition);
		__device__ float2 GradientFluctuatuionAtXT(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize);
		__device__ float2 GradientFluctuatuionAtXZ(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize);

	};



}