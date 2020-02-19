#pragma once

#include <cuda_runtime.h>
#include "..//Options/SolverOptions.h"
#include "..//Options/TurbulentMixingOptions.h"



__device__ inline float linearDissipation(float h, float y, float alpha)
{
	float delta_y = abs(h / 2.0f - y);
	return  -2.0f * alpha * delta_y / h;
}

__device__ inline float linearCreation(float h, float y, float beta)
{
	float delta_y = h / 2.0f - abs(h / 2.0f - y);
	return  2.0f * beta * delta_y / h;
}


__global__ void createTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions);
__global__ void dissipateTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions);
__global__ void advectTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field_0, cudaTextureObject_t v_field_1, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions);