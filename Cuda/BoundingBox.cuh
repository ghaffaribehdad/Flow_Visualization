#pragma once
#include "cuda_runtime.h"
#include "helper_math.cuh"
#include "Ray.cuh"
// 
class BoundingBox
{

private:

	float3 eyePos = { 0,0,0 };
	float3 viewDir = { 0,0,0 };
	float3 upVec = { 0,0,0 };

	float3 nuv[3];

	
	
	float distImagePlane = 0;
	float FOV = 0;
	float aspectRatio = 0;

	float boxFaces[6] = { 0,0,0,0,0,0};
	
	int width = 0;
	int height = 0;

	__host__ __device__ void constructEyeCoordinates();
	__host__ __device__ float3 pixelPosition(const int& i, const int& j);
	__host__ __device__  Ray createRayfromPixel(int i, int j);


	__host__ __device__ float findIntersections(Ray & ray);
	
public:
	__host__ __device__ void setEyePos(const float3 & _eyePos)
	{
		this->eyePos = _eyePos;
	}

	__host__ __device__ void setViewDir(const float3 & _viewDir)
	{
		this->viewDir = _viewDir;
	}

	__host__ __device__ void setUpVect(const float3 _upVec)
	{
		this->upVec = _upVec;
	}

};