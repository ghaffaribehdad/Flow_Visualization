#pragma once
#include "CudaSolver.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

template <class T>
class StreamlineSolver : public CUDASolver<T>
{

public:

	__host__ bool solve();

private:
	__host__ void InitializeVelocityField();
	__host__ void InitializeParticles();
	__host__ void extractStreamlines(Particle<T>* d_particles, VelocityField<T>* d_velocityField);

	Particle<T>* d_particles;

	VelocityField<T> * h_velocityField;
	VelocityField<T> * d_velocityField;
	float3* result;

};

// Kernel of the streamlines
template <class T>
__global__ void TracingParticles(Particle<T>* d_particles, VelocityField<T> * d_velocityField, SolverOptions solverOption, Vertex * p_VertexBuffer);

