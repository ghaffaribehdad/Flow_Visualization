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


class StreaklineSolver : public CUDASolver
{

public:

	__host__ bool solve();
	__host__ bool solveRealtime(int & streakCounter);
	__host__ bool release() override;
	__host__ bool initializeRealtime(SolverOptions * p_solverOptions);
	__host__ virtual bool resetRealtime() override;

private:


	Particle* d_particles;

	float* h_VelocityField;

	VolumeTexture3D volumeTexture_0;
	VolumeTexture3D volumeTexture_1;

	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	// we need three timesteps for RK4




	float3* result;

};

