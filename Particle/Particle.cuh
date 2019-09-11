#pragma once

#include "cuda_runtime.h"
#include "Cuda//CudaHelper.h"

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

	__device__ bool isOut()
	{
		return this->outOfScope;
	}

	__host__ __device__ void  setVelocity(float3& _velocity)
	{
		m_velocity = _velocity;
	}

	// seeding particle
	__host__ void  seedParticle(const float* gridDiameter, const float* seedBox, const float* seedBoxPos);


	__device__ void move(const float& dt, float3 gridDiameter, cudaTextureObject_t t_VelocityField )
	{

		if (!outOfScope)
		{
			this->m_position = RK4EStream(t_VelocityField, &this->m_position, gridDiameter, dt);
		}
		checkPosition(gridDiameter);							//check if it is out of scope
		updateVelocity(gridDiameter, t_VelocityField);//checked
	}


	__device__ void checkPosition(const float3& gridDiameter);

	__device__ void updatePosition(const float dt);


	__device__ void updateVelocity
	(
		const float3& gridDiameter,
		cudaTextureObject_t t_VelocityField
	);

	__device__ __host__ float3 findRelative(const float3& gridDiameter);
	
};