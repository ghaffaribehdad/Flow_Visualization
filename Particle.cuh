#pragma once

#include "cuda_runtime.h"


template <class T>
class Particle
{
private:
	float3 m_position = { 0,0,0 };
	float3 m_velocity = { 0,0,0 };
	bool outOfScope = false;
public:

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

	// seeding particle
	__host__ void  seedParticle(const float3& gridDimenstion);


	__device__ void move(const float& dt, int3 gridSize, float3 gridDiameter, cudaTextureObject_t t_VelocityField )
	{
		updatePosition(dt);										//checked
		checkPosition(gridDiameter);							//check if it is out of scope
		updateVelocity(gridDiameter, gridSize, t_VelocityField);//checked
	}

	__device__ void checkPosition(const float3& gridDiameter);

	__device__ void updatePosition(const float dt);


	__device__ void updateVelocity
	(
		const float3& gridDiameter,
		const int3& gridSize,
		cudaTextureObject_t t_VelocityField
	);

	__device__ __host__ float3 findRelative(const float3& gridDiameter);
	
};