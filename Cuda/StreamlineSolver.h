#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"
#include <vector>
#include <stdio.h>
#include "../Particle/Particle.h"
#include "..//VolumeTexture/VolumeTexture.h"


class StreamlineSolver : public CUDASolver
{

public:

	__host__ bool solve() override;
	__host__ void release();

private:

	Particle* d_particles;
	VolumeTexture volumeTexture;

	float * h_VelocityField;
	float * d_VelocityField;
	
	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	cudaTextureObject_t t_VelocityField = NULL;


	float3* result;

};







