#pragma once
#include <cuda_runtime.h>
#include "../Particle/Particle.h"
#include "../Options/DispresionOptions.h"
#include "../Options/SolverOptions.h"
#include "../Options/FSLEOptions.h"

enum RK4STEP;
enum FTLE_Direction
{
	FORWARD_FTLE = 0,
	BACKWARD_FTLE
};


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
	int timestep,
	unsigned int direction = FTLE_Direction::FORWARD_FTLE
);



__global__ void  traceDispersion3D_path_FSLE
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField_0,
	cudaTextureObject_t velocityField_1,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	FSLEOptions fsleOptions,
	RK4STEP RK4step,
	int timestep
);

__device__ float FTLE3D(Particle* particles, const float & distance);
__device__ float FTLE3D_test(Particle* particles, const float & distance);
__device__ float averageNeighborDistance(Particle* particles);;