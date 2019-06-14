
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

	void initialize();
	void release();

private:

	cudaTextureObject_t t_field;
	float* h_field;

	float3 gridDiameter;
	int3 girdSize;


};