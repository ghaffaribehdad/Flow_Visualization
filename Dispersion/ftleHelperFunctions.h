#pragma once
#include <cuda_runtime.h>
#include "../Particle/Particle.h"
#include "../Options/DispresionOptions.h"
#include "../Options/SolverOptions.h"

enum RK4STEP;


#define FTLE_NEIGHBOR 7 //Number of neighboring particle (6) + the center particle (1)

__global__ void  traceDispersion3D_path_FTLE
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField_0,
	cudaTextureObject_t velocityField_1,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	RK4STEP RK4step,
	int timestep
);

__device__ float FTLE3D(Particle* particles, float distance, float T);