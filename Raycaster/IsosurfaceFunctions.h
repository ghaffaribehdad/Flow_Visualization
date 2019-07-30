#pragma once


#include "cuda_runtime.h"
#include "Raycasting_Helper.h"

class VelocityMagnitude
{

public:

	// calculates the value of the field at position XYZ
	inline __device__ static float ValueAtXYZ(cudaTextureObject_t tex, float3 position)
	{
		return  velocityMagnitude(tex3D<float4>(tex, position.x, position.y, position.z));
	}

	// calculates the gradient of the field at position XYZ
	inline __device__ static float3 GradientAtXYZ(cudaTextureObject_t tex, float3 position, float h)
	{
		float dV_dX = velocityMagnitude(tex3D<float4>(tex, position.x + h/2.0f, position.y, position.z));
		float dV_dY = velocityMagnitude(tex3D<float4>(tex, position.x, position.y + h / 2.0f, position.z));
		float dV_dZ = velocityMagnitude(tex3D<float4>(tex, position.x, position.y, position.z + h / 2.0f));

		dV_dX -= velocityMagnitude(tex3D<float4>(tex, position.x - h / 2.0f, position.y, position.z));
		dV_dY -= velocityMagnitude(tex3D<float4>(tex, position.x, position.y - h / 2.0f, position.z));
		dV_dZ -= velocityMagnitude(tex3D<float4>(tex, position.x, position.y, position.z - h / 2.0f));

		return { dV_dX / h ,dV_dY / h, dV_dZ / h };
	}
};


class Velocity;


class Vorticity;


class WallNormalStress;