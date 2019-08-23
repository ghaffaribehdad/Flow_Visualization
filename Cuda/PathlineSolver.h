#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include "../Particle.cuh"
#include "..//VolumeTexture/VolumeTexture.h"


class PathlineSolver : public CUDASolver
{

public:

	__host__ bool solve();
	__host__ void release();

private:


	Particle<float>* d_particles;


	float* h_VelocityField;
	float* d_VelocityField;

	VolumeTexture volumeTexture_0;
	VolumeTexture volumeTexture_1;

	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	// we need three timesteps for RK4




	float3* result;

};

// Kernel of the streamlines
template <typename T>
__global__ void TracingPath(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step);



