#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"
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

	__host__ void InitializeVelocityField();
	__host__ void InitializeParticles();
	__host__ bool InitializeTexture();

	Particle<T>* d_particles;


	T* h_VelocityField;
	T* d_VelocityField;

	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	cudaTextureObject_t t_VelocityField[3]; // we need three timesteps for RK4

	Particle<T>* d_Particles;
	Particle<T>* h_Particles;

	float3* result;

};

// Kernel of the streamlines
template <typename T>
__global__ void TracingParticles(Particle<T>* d_particles, cudaTextureObject_t  t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer);




