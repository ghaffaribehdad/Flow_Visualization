#pragma once

#include "..//Cuda/helper_math.h"


namespace IsosurfaceHelper
{

	struct Observable
	{
		virtual __device__  float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position);
		virtual __device__  float ValueAtXYZ(cudaTextureObject_t tex, float3 position) { return 0; };
		virtual __device__  float ValueAtXYZ_Surface(cudaSurfaceObject_t tex, float3 position) { return 0; };

		__device__ float3 GradientAtXYZ(cudaTextureObject_t tex, float3 position, int3 gridSize);
		__device__ float3 GradientAtGrid(cudaTextureObject_t tex, float3 relativePos, int3 gridSize);

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

	struct Position : public Observable
	{
		//calculates the value of the field at position XYZ
		__device__ float4 ValueAtXY(cudaTextureObject_t tex, float2 position);
		__device__ float ValueAtXY_Surface_float(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t tex, int3 gridPos);
		__device__ float4 ValueAtXY_Surface_float4(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float2 GradientAtXY_Grid(cudaSurfaceObject_t surf, int2 gridPosition);
		__device__ float2 GradientAtXYZ_Grid(cudaSurfaceObject_t surf, int3 gridPosition);
		__device__ float2 GradientFluctuatuionAtXT(cudaSurfaceObject_t surf, int3 gridPosition);

	};



}