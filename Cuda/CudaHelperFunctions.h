#pragma once

#include <math.h>
#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "../Options/SpaceTimeOptions.h"
#include "../Options/pathSpaceTimeOptions.h"
#include "..//Graphics/Vertex.h"
#include "..//Cuda/helper_math.h"
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


__global__ void TracingPathSurface
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	cudaSurfaceObject_t s_particles,
	SolverOptions solverOptions,
	PathSpaceTimeOptions pathSpaceTimeOptions,
	int step
);

__global__ void initializePathSpaceTime
(
	Particle* d_particle,
	cudaSurfaceObject_t s_pathSpaceTime,
	PathSpaceTimeOptions pathSpaceTimeOptions
);



__global__ void InitializeVertexBufferStreaklines
(
	Particle* d_particles,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
);



//__global__ void applyGaussianFilter
//(
//	int filterSize,
//	int3 FieldSize,
//	cudaTextureObject_t t_velocityField,
//	cudaSurfaceObject_t	s_velocityField
//);



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


inline __device__ __host__ int3 indexto3D(int3 gridSize, int index) {
	int3 index3D = make_int3(0, 0, 0);
	index3D.x = int(index / (gridSize.y * gridSize.z));
	index3D.y = int((index - index3D.x*gridSize.y*gridSize.z) / gridSize.z);
	index3D.z = index - index3D.x*gridSize.y*gridSize.z - index3D.y*gridSize.z;

	return index3D;
};


inline __device__ __host__ float depthfinder(const float3& position, const float3& eyePos, const float3& viwDir, const float& f, const float& n)
{

	// calculates the z-value
	double z_dist = abs(dot(viwDir, position - eyePos));
	double d_f = f;
	double d_n = n;
	// calculate non-linear depth between 0 to 1
	double depth = (d_f) / (d_f - d_n);
	depth += (-1.0 / z_dist) * (d_f * d_n) / (d_f - d_n);

	return static_cast<float>(depth);
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


__device__  inline float4 filterGaussian2D(int filterSize, float std, cudaTextureObject_t tex, int direction, float3 position)
{
	float4 filteredValue = { 0,0,0,0 };
	switch (direction)
	{

	case 0: //XY
	{

		break;
	}

	case 1: //YZ
	{

		break;
	}

	case 2: //ZX
	{

		for (int ii = 0; ii < filterSize; ii++)
		{
			for (int jj = 0; jj < filterSize; jj++)
			{
				filteredValue = filteredValue +
					0.5 * (1 / CUDA_PI_D) * (1.0 / powf(std, 2)) * exp(-1.0 * ((powf(ii - filterSize / 2, 2) + powf(jj - filterSize / 2, 2)) / (2.0f * std * std)))*
					tex3D<float4>(tex, position.x + jj - filterSize / 2, position.y, position.z + ii - filterSize / 2);
			}
		}
		break;
	}

	}

	return filteredValue;
}



