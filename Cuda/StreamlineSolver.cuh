#pragma once
#include "CudaSolver.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>


class StreamlineSolver : public CUDASolver
{

public:

	__host__ bool InitializeCUDA();



private:

	Particle* d_particles;
	VelocityField * h_velocityField;
	VelocityField * d_velocityField;

};

__global__ void TracingParticles(Particle* d_particles, VelocityField* d_velocityField);