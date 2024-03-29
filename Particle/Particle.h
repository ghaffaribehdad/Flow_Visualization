#pragma once

#include "cuda_runtime.h"

struct Particle
{

	float3 m_position = { 0,0,0 };
	float3 m_velocity = { 0,0,0 };

	__device__ void updatePosition(const float dt);
	__device__ void updateVelocity(const float3& gridDiameter, const int3& gridSize, cudaTextureObject_t t_VelocityField, float3 velocityScale = { 1.0f,1.0f ,1.0f });
	__device__ __host__ float3 findRelative(const float3& gridDiameter);

	__device__ __host__ Particle()
	{
		m_position = { 0,0,0 };
		m_velocity = { 0,0,0 };
	}

	// Setter and Getter functions
	__device__ __host__ float3* getPosition()
	{
		return & m_position;
	}
	__host__ __device__ void setPosition(float3& _position)
	{
		m_position = _position;
	}

	__host__ __device__ float3*  getVelocity()
	{
		return &m_velocity;
	}


	__host__ __device__ void  setVelocity(float3& _velocity)
	{
		m_velocity = _velocity;
	}

	bool diverged = false;
	float fsle = 0;
	
};