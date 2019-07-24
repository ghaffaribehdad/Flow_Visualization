#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include "../Particle.cuh"



template <class T>
class PathlineSolver : public CUDASolver<T>
{

public:

	__host__ bool solve();
	__host__ void release();

private:


	Particle<T>* d_particles;


	T* h_VelocityField;
	T* d_VelocityField;

	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	// we need three timesteps for RK4
	cudaTextureObject_t t_VelocityField_0;
	cudaTextureObject_t t_VelocityField_1;



	float3* result;

};

// Kernel of the streamlines
template <typename T>
__global__ void TracingPath(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step);




