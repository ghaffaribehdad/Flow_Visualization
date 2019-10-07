#pragma once

#include "../Particle/Particle.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"

void seedParticle_ZY_Plane(Particle* particle, const float* gridDiameter, const int* gridSize, const float& y_slice);

__global__ void traceDispersion
(
	int timeStep,
	float dt,
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface,
	cudaTextureObject_t velocityField,
	int nParticles,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
);
