
#pragma once
#include "cuda_runtime.h"

class VolumeTexture
{

public:

	VolumeTexture();

	// setter functions
	void setGridDiameter(const float3& _gridDiamter);
	void setGridSize(const int3& _gridSize);

	void setField(float* _h_field);

	const int3& getGridSize() const;
	const float3& getGridDiameter() const;

	void initialize();
	void release();


	__device__ float4 fetch(float3 index);
	

private:

	cudaTextureObject_t t_field;
	float* h_field;

	float3 gridDiameter;
	int3 gridSize;



};