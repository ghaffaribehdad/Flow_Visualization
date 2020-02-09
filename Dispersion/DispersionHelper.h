#pragma once

#include "../Particle/Particle.h"

#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Options/SolverOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"

void seedParticle_ZY_Plane(Particle* particle, float* gridDiameter, const int* gridSize, const float& y_slice);
void seedParticle_tiltedPlane(Particle* particle, float* gridDiameter, const int* gridSize, const float& y_slice, const float & tilt);


enum RK4STEP
{
	EVEN = 0,
	ODD
};

__global__ void traceDispersion
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
);


__global__ void traceDispersion3D
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
);


__global__ void traceDispersion3D_extra
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
);

__global__ void traceDispersion3D_path
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField_0,
	cudaTextureObject_t velocityField_1,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	RK4STEP RK4step,
	int timeStep
);

//__global__ void trace_fluctuation3D
//(
//	cudaSurfaceObject_t heightFieldSurface3D,
//	cudaSurfaceObject_t heightFieldSurface3D_extra,
//	cudaTextureObject_t velocityField_0,
//	SolverOptions solverOptions,
//	FluctuationheightfieldOptions fluctuationOptions,
//	int timestep
//);





template <typename Observable>
__global__ void heightFieldGradient
(

	cudaSurfaceObject_t heightFieldSurface,
	cudaSurfaceObject_t heightFieldSurface_gradient,
	DispersionOptions dispersionOptions,
	SolverOptions	solverOptions
);

template <typename Observable>
__global__ void heightFieldGradient3D
(

	cudaSurfaceObject_t heightFieldSurface3D,
	DispersionOptions dispersionOptions,
	SolverOptions	solverOptions
);


template <typename Observable>
__global__ void fluctuationfieldGradient3D
(
	cudaSurfaceObject_t heightFieldSurface3D,
	SolverOptions solverOptions,
	FluctuationheightfieldOptions fluctuationOptions

);


