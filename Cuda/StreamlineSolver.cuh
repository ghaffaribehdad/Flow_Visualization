#pragma once
#include "CudaSolver.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>


class StreamlineSolver : public CUDASolver
{

public:

	__host__ bool solve();

private:
	__host__ void InitializeVelocityField();
	__host__ void InitializeParticles();
	__host__ void extractStreamlines();

	Particle* d_particles;

	VelocityField * h_velocityField;
	VelocityField * d_velocityField;
	float3* result;

};

// Kernel of the streamlines

__global__ void TracingParticles(Particle* d_particles, VelocityField* d_velocityField, SolverOptions solverOption, Vertex * p_VertexBuffer);