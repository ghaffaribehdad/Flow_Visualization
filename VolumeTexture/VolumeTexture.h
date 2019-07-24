
#pragma once
#include "cuda_runtime.h"
#include "..//ErrorLogger.h"
#include "texture_fetch_functions.h"

class VolumeTexture
{

public:

	// setter functions
	void setGridDiameter(const float3& _gridDiamter);
	void setGridSize(const int3& _gridSize);

	void setField(float* _h_field);

	const int3& getGridSize() const;
	const float3& getGridDiameter() const;

	void initialize();
	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}
	

private:

	cudaTextureObject_t t_field;
	float* h_field;

	float3 gridDiameter;
	int3 gridSize;



};