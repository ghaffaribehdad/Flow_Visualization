#pragma once

#include "..//Cuda/helper_math.h"


namespace IsosurfaceHelper
{

	struct Observable
	{
		virtual __device__  float ValueAtXYZ(cudaTextureObject_t tex, float3 position) { return 0; };
		__device__  float3 GradientAtXYZ(cudaTextureObject_t tex, float3 position, float h);
	
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


}
