#pragma once
#include "CudaSolver.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

template <class T>
class StreamlineSolver : public CUDASolver<T>
{

public:

	__host__ bool solve();

private:

	__host__ T* InitializeVelocityField();
	__host__ void InitializeParticles();
	__host__ bool InitializeTexture(T* h_VelocityField);

	Particle<T>* d_particles;


	T * h_VelocityField;
	T * d_VelocityField;
	
	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 
	cudaTextureObject_t t_VelocityField = NULL;

	Particle<T>* d_Particles;
	Particle<T>* h_Particles;
	float3* result;

};

// Kernel of the streamlines
template <typename T>
__global__ void TracingParticles(Particle<T>* d_particles, T* d_velocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int timesteps = solverOptions.timestep;
	int index_buffer = index * timesteps;
	float dt = solverOptions.dt;
	float3 gridDiameter =
	{
		solverOptions.gridDiameter[0],
		solverOptions.gridDiameter[1],
		solverOptions.gridDiameter[2]
	};

	int3 gridSize =
	{
		solverOptions.gridSize[0],
		solverOptions.gridSize[1],
		solverOptions.gridSize[2]
	};

	for (int i = 0; i < timesteps; i++)
	{
		d_particles[index].move(dt, d_velocityField,gridSize,gridDiameter);

		p_VertexBuffer[index_buffer +i].pos.x = d_particles[index].getPosition()->x;
		p_VertexBuffer[index_buffer +i].pos.y = d_particles[index].getPosition()->y;
		p_VertexBuffer[index_buffer +i].pos.z = d_particles[index].getPosition()->z;

		p_VertexBuffer[index_buffer +i].texCoord.x = index;
		p_VertexBuffer[index_buffer + i].texCoord.y = index;
		if (i == timesteps - 1)
		{
			p_VertexBuffer[index_buffer + i].texCoord.y = -1;
		}
	}


}


// Kernel of the streamlines
template <typename T>
__global__ void TracingParticles(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
{
	//tex3Dfetch(t_VelocityField, 0, 0, 0);

}
