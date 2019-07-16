#pragma once
#include "cuda_runtime.h"
#include "helper_math.cuh"

class Ray
{
private:

	float3 eyePos;
	float3 pixelPos;

public:

	__host__ __device__ Ray(const float3 & _eyePos, const float3 & _pixelPos):
		eyePos(_eyePos),pixelPos(_pixelPos)
	{}

	__host__ __device__ void setEyePos(const float3& _eyePos)
	{
		this->eyePos = _eyePos;
	}

	__host__ __device__ void setPixelPos(const float3& _pixelPos)
	{
		this->pixelPos = _pixelPos;
	}

	__host__ __device__ float3 getPos(const float& t)
	{
		float3 pos = eyePos + t * (pixelPos - eyePos);

		return pos;
	}

	__host__ __device__ float3 & getPixelPos()
	{
		return this->pixelPos;
	}

};