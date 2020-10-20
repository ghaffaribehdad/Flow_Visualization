#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include "../Particle/Particle.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Timer/Timer.h"


class PathlineSolver : public CUDASolver
{

public:

	__host__ bool solve();
	__host__ bool solveRealtime();
	__host__ bool initializeRealtime();
	__host__ void release();

private:

	Timer timer;
	uint timeSteps = 0;
	Particle* d_particles = nullptr;

	float* h_VelocityField = nullptr;
	float* d_VelocityField = nullptr;

	VolumeTexture3D volumeTexture_0;
	VolumeTexture3D volumeTexture_1;

	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	// we need three timesteps for RK4

	

	float3* result;

};




