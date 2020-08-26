#pragma once

#include <math.h>
#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "..//Graphics/Vertex.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "cuda_runtime.h"



__device__ void	RK4Path
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity,
	float3 velocityScale = { 1.0f,1.0f,1.0f }
);


// Switch the velocity texture for even and odd case
__device__ void	RK4Path_linear
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity
);



//__device__ void	Euler_2D
//(
//	const int2& initialGridPosition,
//	float2& finalGridPosition,
//	const int2& gridSize,
//	const float2& gridDiameter,
//	const float& dt,
//	cudaTextureObject_t t_VelocityField_0
//);



__device__ void RK4Stream
(
	cudaTextureObject_t t_VelocityField_0,
	Particle* particle,
	const float3& gridDiameter,
	const int3& gridSize,
	float dt,
	float3 velocityScale = { 1.0f,1.0f,1.0f }
);



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

//__global__ void TracingStream
//(
//	Particle* d_particles,
//	cudaTextureObject_t t_VelocityField,
//	SolverOptions solverOptions,
//	Vertex* p_VertexBuffer,
//	float4* d_VertexBuffer
//);
//__global__ void TracingStream
//(
//	Particle* d_particles,
//	cudaTextureObject_t t_VelocityField,
//	cudaTextureObject_t t_Vorticity,
//	SolverOptions solverOptions,
//	Vertex* p_VertexBuffer,
//	float4* d_VertexBuffer
//);
//
//__global__ void Vorticity
//(
//	cudaTextureObject_t t_VelocityField,
//	SolverOptions solverOptions,
//	cudaSurfaceObject_t	s_measure
//);





//template <typename Observable>
//__device__ float3 binarySearch
//(
//	Observable& observable,
//	cudaTextureObject_t field,
//	float3& _position,
//	float3& gridDiameter,
//	int3& gridSize,
//	float3& _samplingStep,
//	float& value,
//	float& tolerance,
//	int maxIteration
//);


template <typename Observable>
__device__ float3 binarySearch
(
	Observable& observable,
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	int3& gridSize,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
)
{
	float3 position = _position;
	float3 relative_position = world2Tex(position, gridDiameter, gridSize);
	float3 samplingStep = _samplingStep * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;

	while (fabsf(observable.ValueAtXYZ_Tex(field, relative_position) - value) > tolerance&& counter < maxIteration)
	{

		if (observable.ValueAtXYZ_Tex(field, relative_position) - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			position = position - samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			position = position + samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 1;

		}
		counter++;

	}

	return position;

};



__device__ float3 binarySearch_X
(
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
);


inline __device__ __host__ int2 index2pixel(const int& index, const int& width)
{
	int2 pixel = { 0,0 };
	pixel.y = index / width;
	pixel.x = index - pixel.y * width;

	return pixel;
};



inline __device__ __host__ float depthfinder(const float3& position, const float3& eyePos, const float3& viwDir, const float& f, const float& n)
{

	// calculates the z-value
	float z_dist = abs(dot(viwDir, position - eyePos));

	// calculate non-linear depth between 0 to 1
	float depth = (f) / (f - n);
	depth += (-1.0f / z_dist) * (f * n) / (f - n);

	return depth;
};

__device__ inline float getTemperature(float* temp, float pos, int size, int offset)
{
	pos = pos - offset;

	if (ceil(pos) < size && pos > 0)
	{
		if ((int)pos == pos)
		{
			return temp[(int)pos];
		}
		else
		{
			return (pos - floor(pos)) * temp[(int)ceil(pos)] + (ceil(pos) - pos) * temp[(int)floor(pos)];

		}

	}
	else if (pos < 0)
	{
		//return temp[0];
		return 0.0f; // Border Address Mode
		
	}
	else
	{
		//return temp[size - 1];
		return 0.0f; // Border Address Mode
	}
}





