
#pragma once
#include "cuda_runtime.h"
#include "../ErrorLogger/ErrorLogger.h"
#include "texture_fetch_functions.h"
#include "../Options/SolverOptions.h"

class VolumeTexture
{

public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}

	void setSolverOptions(SolverOptions* _solverOptions)
	{
		this->solverOptions = _solverOptions;
	}


	// Create a texture and populate it with h_field
	// Address modes can be set for X,y,z
	cudaTextureObject_t initialize
	(
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap
	);

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}
	

private:

	SolverOptions * solverOptions;

	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;

	float* h_field;

	float3 gridDiameter;



};