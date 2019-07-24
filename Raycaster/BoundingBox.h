#pragma once
#include "cuda_runtime.h"
#include <string>
#include "..//Cuda/helper_math.h"



struct BoundingBox
{
public:

	float3 eyePos;
	float3 viewDir;
	float3 upVec;
	float3 nuv[3];


	float distImagePlane;
	float FOV;
	float aspectRatio;
	float3 gridDiameter;

	float boxFaces[6];

	int width;
	int height;



	__host__ __device__ void constructEyeCoordinates();
	__host__ __device__ void updateBoxFaces();
	__host__ __device__ void updateAspectRatio();
};