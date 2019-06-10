#pragma once

#include "cuda_runtime.h"
#include "../linearIndex.cuh"
#include "VelocityField.cuh"

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
	__host__ void  seedParticle(const float3& gridDimenstion)
	{
		this->m_position.x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.x);
		this->m_position.y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.y);
		this->m_position.z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.z);

	}


	__device__ void move(const float& dt, T * d_velocityField, int3 gridSize, float3 gridDiameter )
	{
;

		updatePosition(dt);										//checked
		checkPosition(gridDiameter);							//check if it is out of scope
		updateVelocity(gridDiameter, gridSize, d_velocityField);//checked
	}

	__device__ void checkPosition(const float3 & gridDiameter)
	{
		
		if (m_position.x >= gridDiameter.x)
		{
			m_position.x = gridDiameter.x;
			this->outOfScope = true;
		}
		if (m_position.y >= gridDiameter.y)
		{
			m_position.y = gridDiameter.y;
			this->outOfScope = true;
		}
		if (m_position.z >= gridDiameter.z)
		{
			m_position.z = gridDiameter.z;
			this->outOfScope = true;
		}

	}
	__device__ void updatePosition(const float dt)
	{
		if (!outOfScope)
		{
			this->m_position.x += dt * (this->m_velocity.x);
			this->m_position.y += dt * (this->m_velocity.y);
			this->m_position.z += dt * (this->m_velocity.z);
		}

	}


	__device__ void updateVelocity(const float3 & gridDiameter, const int3 & gridSize, T * p_velocityField)
	{
		if (!outOfScope)
		{
			int3 index = findIndex(gridDiameter, gridSize);

			int4 dim = { gridSize.x,gridSize.y,gridSize.z,3 };
			uint mappedIndex_Vx = linearIndex(index.x, index.y, index.z, 0, dim);
			T v_x = p_velocityField[mappedIndex_Vx];
			T v_y = p_velocityField[mappedIndex_Vx + 1];
			T v_z = p_velocityField[mappedIndex_Vx + 2];

			//TO-DO:: Correct the next line
			float3 velocity = { static_cast<float>(v_x), static_cast<float>(v_y), static_cast<float>(v_z) };
			this->setVelocity(velocity);
		}

	}
	__device__ __host__ int3 findIndex(const float3 & gridDiameter, const int3 & gridSize)
	{
		float3 relative_position = {
			(static_cast<float>(this->m_position.x)) / (gridDiameter.x),
			(static_cast<float>(this->m_position.y)) / (gridDiameter.y),
			(static_cast<float>(this->m_position.z)) / (gridDiameter.z)
		};
		int3 index = { 0,0,0 };
		index.x = static_cast<int>(relative_position.x * (gridSize.x - 1));
		index.y = static_cast<int>(relative_position.y * (gridSize.y - 1));
		index.z = static_cast<int>(relative_position.z * (gridSize.z - 1));

		return index;
	}
};