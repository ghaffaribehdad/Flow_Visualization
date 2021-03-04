#pragma once

#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "..//Graphics/Vertex.h"
#include "cuda_runtime.h"

namespace ParticleTracing
{
	__device__ float3 RK4
	(
		cudaTextureObject_t t_VelocityField_0,
		cudaTextureObject_t t_VelocityField_1,
		const float3 & position,
		const float3 & gridDiameter,
		const int3 & gridSize,
		const float & dt,
		const float3 & velocityScale = { 1.0f,1.0f,1.0f }
	);

	__global__ void TracingStream
	(
		Particle* d_particles,
		cudaTextureObject_t t_VelocityField,
		SolverOptions solverOptions,
		Vertex* p_VertexBuffer
	);

	__device__ void RK4Stream(
		cudaTextureObject_t t_VelocityField_0,
		Particle* particle,
		const float3& gridDiameter,
		const int3& gridSize,
		float dt,
		float3 velocityScale
	);

	template <typename measure>
	__device__ float colorCode(cudaTextureObject_t t_VelocityField_0, float3 texPos)
	{
		
	}
}

