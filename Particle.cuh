#pragma once

#include "cuda_runtime.h"
#include "../linearIndex.cuh"
#include "VelocityField.cuh"


class Particle
{
private:
	float3 m_position = { 0,0,0 };
	float3 m_velocity = { 0,0,0 };
	bool outOfScope = false;
public:

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
		m_position = _velocity;
	}

	// seeding particle
	__host__ void  seedParticle(const float3& gridDimenstion)
	{
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.x);
		float y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.y);
		float z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.z);
	}


	__device__ __host__ void move(const float& dt, VelocityField * p_velocityField)
	{

		float3 gridDiameter = p_velocityField->getGridDiameter();
		int3 gridSize = p_velocityField->getGridSize();

		//this->updatePosition(dt);						//checked
		//this->updateVelocity(gridDiameter, gridSize, p_velocityField);	//checked
		//this->checkPosition(gridDiameter);			//check if it is out of scope
	}

	__device__ __host__ void checkPosition(const float3 & gridDiameter)
	{
		
		if (m_position.x >= gridDiameter.x)
		{
			this->outOfScope = true;
		}
		if (m_position.y >= gridDiameter.y)
		{
			this->outOfScope = true;
		}
		if (m_position.z >= gridDiameter.z)
		{
			this->outOfScope = true;
		}

	}
	__device__ __host__ void updatePosition(const float dt)
	{
		if (!outOfScope)
		{
			this->m_position.x += dt * (this->m_velocity.x);
			this->m_position.y += dt * (this->m_velocity.y);
			this->m_position.z += dt * (this->m_velocity.z);
		}

	}

	__device__ __host__ void updateVelocity(const float3 & gridDiameter, const int3 & gridSize, VelocityField* p_velocityField)
	{
		int3 index = findIndex(gridDiameter, gridSize);

		int3 dim = { gridSize.x,gridSize.y,gridSize.z};
		uint mappedIndex_Vx = linearIndex(index.x, index.y, index.z, dim);
		float v_x = p_velocityField->getVelocityField()[mappedIndex_Vx];
		float v_y = p_velocityField->getVelocityField()[mappedIndex_Vx + 1];
		float v_z = p_velocityField->getVelocityField()[mappedIndex_Vx + 2];
		float3 velocity = { v_x, v_y, v_z };
		this->setVelocity(velocity);
	}
	__device__ __host__ int3 findIndex(const float3 & gridDiameter, const int3 & gridSize)
	{
		float3 relative_position = {
			(this->m_position.x) / (gridDiameter.x),
			(this->m_position.y) / (gridDiameter.y),
			(this->m_position.z) / (gridDiameter.z)
		};
		int3 index = { 0,0,0 };
		index.x = static_cast<int>(relative_position.x * (gridSize.x - 1));
		index.y = static_cast<int>(relative_position.y * (gridSize.y - 1));
		index.z = static_cast<int>(relative_position.z * (gridSize.z - 1));

		return index;
	}
};