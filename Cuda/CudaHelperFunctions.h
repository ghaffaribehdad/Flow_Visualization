#pragma once

#include <math.h>
#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "../Options/fluctuationheightfieldOptions.h"
#include "..//Graphics/Vertex.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "cuda_runtime.h"

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

__device__ void	RK4Path
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	float3 velocityScale = { 1.0f,1.0f,1.0f }
);



__device__ Particle RK4Streak
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	float3 position,
	float3 gridDiameter,
	int3 gridSize,
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



__global__ void InitializeVertexBufferStreaklines
(
	Particle* d_particles,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
);

template <typename T1, typename T2, typename T3, typename T4>
__global__ void copyTextureToSurface
(
	int streamwisePos,
	int time,
	SolverOptions solverOptions,
	cudaTextureObject_t t_velocityField,
	cudaSurfaceObject_t	s_velocityField
)
{
	int index = CUDA_INDEX; // The Spanwise Position

	float textureOffset = 0.5;

	if (index < solverOptions.gridSize[2])
	{

		for (int y = 0; y < solverOptions.gridSize[1]; y++)
		{
			T1 measure1;
			T2 measure2;
			T3 measure3;
			T4 measure4;

			float3 pos = make_float3(streamwisePos + textureOffset, y + textureOffset, index + textureOffset);
			float value1 = measure1.ValueAtXYZ_Tex(t_velocityField, pos);
			float value2 = measure2.ValueAtXYZ_Tex(t_velocityField, pos);
			float value3 = measure3.ValueAtXYZ_Tex(t_velocityField, pos);
			float value4 = measure4.ValueAtXYZ_Tex(t_velocityField, pos);

			float4 value = make_float4(value1, value2, value3, value4);

			surf3Dwrite(value, s_velocityField, 4 * sizeof(float) * time, y,  index);

		}
	}


}

__global__ void applyGaussianFilter
(
	int filterSize,
	int3 FieldSize,
	cudaTextureObject_t t_velocityField,
	cudaSurfaceObject_t	s_velocityField
);


__global__ void AddOffsetVertexBufferStreaklines
(
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
);

__global__ void TracingStreak
(
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

template <typename Observable>
__device__ float3 binarySearch_GradientBased
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

	while (fabsf(observable.ValueAtXYZ_Tex_GradientBase(field, relative_position, gridDiameter, gridSize) - value) > tolerance&& counter < maxIteration)
	{

		if (observable.ValueAtXYZ_Tex_GradientBase(field, relative_position, gridDiameter, gridSize) - value > 0)
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


__device__ inline float* gaussianFilter2D( int size, float std = 1);
__device__ inline void applyFilter2D(float * filter, int size, cudaTextureObject_t tex, cudaSurfaceObject_t surf, int direction, int plane, int3 gridSize);
__device__ inline void gaussianFilter3D(cudaTextureObject_t tex, cudaSurfaceObject_t surf, int3 size);



