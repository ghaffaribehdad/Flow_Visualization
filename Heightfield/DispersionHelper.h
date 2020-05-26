#pragma once

#include "../Particle/Particle.h"
#include "../Particle/ParticleHelperFunctions.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Options/SolverOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"



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



template <typename Observable>
__global__ void heightFieldGradient3D
(

	cudaSurfaceObject_t heightFieldSurface3D,
	DispersionOptions dispersionOptions,
	SolverOptions	solverOptions
);


__global__ void heightFieldGradient3DFTLE
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

__global__ void textureMean
(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_mean_height,
	float * d_mean_ftle,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
);

__global__ void fetch_ftle_height
(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_height,
	float * d_ftle,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	int timestep
);


__global__ void pearson_terms
(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_mean_height,
	float * d_mean_ftle,
	float * d_pearson_cov,
	float * d_pearson_var_ftle,
	float * d_pearson_var_height,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
);

__global__ void pearson
(
	float * d_pearson_cov,
	float * d_pearson_var_ftle,
	float * d_pearson_var_height,
	SolverOptions solverOptions
);

