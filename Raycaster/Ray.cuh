#pragma once
#include "cuda_runtime.h"
#include "BoundingBox.cuh"

class Ray
{
private:
	
	float3 eyePos = { 0,0,0 };
	float3 pixelPos = { 0,0,0 };

	float2 tNearFar = { -1,-1 };

	float3 farIntersection = { 0,0,0 };
	float3 nearIntersection = { 0,0,0 };



public:
	int2 pixel = { 0,0 };
	// calculate bounding box intersection with the ray (tNear and tFar)
	__device__  void boundingBoxIntersection(BoundingBox* boundingBox)
	{
		this->pixelPos = boundingBox->pixelPosition(this->pixel.x, this->pixel.y);
		this->tNearFar = boundingBox->findIntersections(this->pixelPos);
	}


	__device__ void extractIntersection()
	{
		this->nearIntersection = eyePos + this->tNearFar.x * (pixelPos - eyePos);
		this->farIntersection = eyePos + this->tNearFar.y * (pixelPos - eyePos);
	}


	__host__ __device__ float2 getIntersection()
	{
		return this->tNearFar;
	}



	__host__ __device__ Ray(const float3& _eyePos, const int2& _pixel) :
		eyePos(_eyePos), pixel(_pixel)
	{}

	__host__ __device__ Ray(const float3& _eyePos) :
		eyePos(_eyePos)
	{}

	__host__ __device__ Ray()
	{}

	__host__ __device__ void setEyePos(const float3& _eyePos)
	{
		this->eyePos = _eyePos;
	}

	__host__ __device__ void setPixelPos(const float3& _pixelPos)
	{
		this->pixelPos = _pixelPos;
	}



	__host__ __device__ float3& getPixelPos()
	{
		return this->pixelPos;
	}

	__host__ __device__ void setPixel(const int& i, const int& j)
	{
		this->pixel.x = i;
		this->pixel.y = j;
	}


};