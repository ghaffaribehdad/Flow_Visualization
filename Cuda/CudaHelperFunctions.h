#pragma once
#include "cuda_runtime.h"
#include <math.h>
#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "..//Graphics/Vertex.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "..//Cuda/helper_math.h"



// Switch the velocity texture for even and odd case
__device__ void	RK4Path	
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,	
	float3 gridDiameter,	
	float dt,
	bool periodicity
);

__device__ void RK4Stream(cudaTextureObject_t t_VelocityField_0, Particle* particle, float3 gridDiameter, float dt);

__host__ void seedParticleGridPoints(Particle* particle, const SolverOptions* solverOptions);

__host__ void  seedParticleRandom(Particle* particle, const SolverOptions* solverOptions);

__global__ void TracingPath
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer,
	bool odd,
	int step
);


__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
);

__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer,
	float4* d_VertexBuffer
);
__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	cudaTextureObject_t t_Vorticity,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer,
	float4* d_VertexBuffer
);

__global__ void Vorticity
(
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	cudaSurfaceObject_t	s_measure
);


template <typename Observable>
__device__ float3 binarySearch
(
	Observable & observable,
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
)
{
	float3 position = _position;
	float3 relative_position = position / gridDiameter;
	float3 samplingStep = _samplingStep*0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;

	while (fabsf(observable.ValueAtXYZ(field, relative_position) - value) > tolerance && counter < maxIteration)
	{

		if (observable.ValueAtXYZ(field, relative_position) - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			position = position - samplingStep;
			relative_position = position / gridDiameter;
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			
			position = position + samplingStep;
			relative_position = position / gridDiameter;
			side = 1;
			
		}
		counter++;
		
	}

	return position;

};


template <typename Observable>
__device__ float3 binarySearchHeightField
(
	Observable& observable,
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	float3& _samplingStep,
	float& tolerance,
	int maxIteration
)
{
	
	float3 position = _position;
	float3 relative_position = position / gridDiameter;
	float3 samplingStep = _samplingStep * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;
	float value = position.y;
	while (fabsf(observable.ValueAtXY(field, relative_position).y - value) > tolerance && counter < maxIteration)
	{

		if (observable.ValueAtXY(field, relative_position).y - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			// return position if we are out of texture
			if (outofTexture((position + samplingStep)/ gridDiameter))
				return position;

			position = position - samplingStep;
			relative_position = position / gridDiameter;
			
			value = position.y;
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			// return position if we are out of texture
			if (outofTexture((position + samplingStep) / gridDiameter))
				return position;
			
			position = position + samplingStep;
			relative_position = position / gridDiameter;
			value = position.y;
			side = 1;

		}
		counter++;

	}

	return position;

};




__device__ float3 binarySearch_heightField
(
	float3 _position,
	cudaSurfaceObject_t tex,
	float3 _samplingStep,
	float3 gridDiameter,
	float tolerance,
	int maxIteration
);


