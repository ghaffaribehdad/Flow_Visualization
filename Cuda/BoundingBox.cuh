#pragma once
#include "cuda_runtime.h"
#include "helper_math.cuh"
#include <string>


class BoundingBox
{

private:

	float3 eyePos = { 0.0f, 0.0f, 0.0f };
	float3 viewDir = { 0.0f,0.0f,1.0f };
	float3 upVec = { 0.0f,1.0f,0.0f };

	float3 nuv[3];


	float distImagePlane = 1;
	float FOV = 90 / 180.0f * 3.14159265f;
	float aspectRatio = 1.0f;
	float3 gridDiameter = { 0,0,0 };

	float boxFaces[6] = { -1.0,1.0,-1.0,1.0,6,5 };

	int width = 0;
	int height = 0;



	__device__ __host__ void constructEyeCoordinates();
	__host__ __device__ void updateBoxFaces();
	__host__ __device__ void updateAspectRatio();


public:

	// Initialize BoundingBox rendering
	__device__ __host__ void initialize();

	// return tNear and tFar for pixel i and j
	__device__ float2 findIntersections(float3& pixelPos);

	// compute pixelPosition for pixel i and j
	__device__ float3 pixelPosition(const int& i, const int& j);

	// setter and getter functions
	__host__ void setEyePos(const float3& _eyePos)
	{
		this->eyePos = _eyePos;
	}

	__host__ void setGridDiametr(const float3& _gridDiameter)
	{
		this->gridDiameter = _gridDiameter;
	}
	__host__ void setAspectRatio(const float& _aspectRatio)
	{
		this->aspectRatio = _aspectRatio;
	}

	__host__ void setViewDir(const float3& _viewDir)
	{
		this->viewDir = _viewDir;
	}

	__host__ void setUpVect(const float3 _upVec)
	{
		this->upVec = _upVec;
	}


	__host__ void setResolution(const int& _width, const int& _height)
	{
		this->width = _width;
		this->height = _height;
	}

	__host__ __device__ int2 getResolution()
	{
		return { this->width, this->height };
	}

};