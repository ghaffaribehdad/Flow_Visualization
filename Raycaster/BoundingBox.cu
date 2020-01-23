#include "BoundingBox.h"
#include <iostream>
#include <fstream>
#include "../Cuda/helper_math.h"

__device__ __host__ void BoundingBox::constructEyeCoordinates(const float3& eyePos, const float3& viewDir, const float3& upVec)
{

	this->m_eyePos = eyePos;
	this->m_viewDir = viewDir;
	this->m_upVec = upVec;

	// N vector
	this->nuv[0] = normalize(this->m_viewDir);

	// U vector
	float3 upVectNorm = normalize(this->m_upVec);
	this->nuv[1] = cross(nuv[0], upVectNorm);

	// V vector
	this->nuv[2] = cross(nuv[0], nuv[1]);
}


__host__ __device__ void BoundingBox::updateBoxFaces(const float3 & dimensions)
{
	m_dimensions = dimensions;
	this->boxFaces[0] = dimensions.x / -2.0f;
	this->boxFaces[1] = dimensions.x / 2.0f;

	this->boxFaces[2] = dimensions.y / -2.0f;
	this->boxFaces[3] = dimensions.y / 2.0f;

	this->boxFaces[4] = dimensions.z / -2.0f;
	this->boxFaces[5] = dimensions.z / 2.0f;


}

__host__ __device__ void BoundingBox::updateAspectRatio(const int & width, const int & height)
{
	this->m_width = width;
	this->m_height = height;
	this->aspectRatio = static_cast<float>(this->m_width) / static_cast<float>(this->m_height);
}