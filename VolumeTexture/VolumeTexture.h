
#pragma once
#include "cuda_runtime.h"
#include "..//ErrorLogger.h"
#include "texture_fetch_functions.h"
#include "..//SolverOptions.h"

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


	cudaTextureObject_t initialize();

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}
	

private:

	SolverOptions * solverOptions;

	cudaTextureObject_t t_field;

	float* h_field;

	float3 gridDiameter;



};