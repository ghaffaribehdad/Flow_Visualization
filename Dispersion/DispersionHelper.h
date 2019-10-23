#pragma once

#include "../Particle/Particle.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"

void seedParticle_ZY_Plane(Particle* particle, float* gridDiameter, const int* gridSize, const float& y_slice);

__global__ void traceDispersion
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
);

template <typename Observable>
__global__ void hieghtfieldGradient
(

	cudaSurfaceObject_t heightFieldSurface,
	cudaSurfaceObject_t heightFieldSurface_gradient,
	DispersionOptions dispersionOptions,
	SolverOptions	solverOptions
);