#pragma once
#include "cuda_runtime.h"
#include <string>
#include <DirectXMath.h>
#define _XM_NO_INTRINSICS_



struct BoundingBox
{
public:

	float3 m_eyePos;
	float3 m_viewDir;
	float3 m_upVec;
	float3 nuv[3];


	float distImagePlane;
	float FOV;
	float aspectRatio;
	int3 gridSize;
	float3 gridDiameter;
	float3 m_clipBox;
	float boxFaces[6];

	int m_width;
	int m_height;


	__host__ __device__ void constructEyeCoordinates(const float3& eyePos, const float3& viewDir, const float3& upVec);
	__host__ __device__ void updateBoxFaces(const float3 & dimensions, const float3 & center);
	__host__ __device__ void updateAspectRatio(const int & width, const int & height);
};