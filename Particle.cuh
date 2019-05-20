#pragma once

#include "cuda_runtime.h"
#include "../linearIndex.cuh"
#include "VelocityField.h"


class Particle
{
private:
	float3 m_position = { 0,0,0 };
	float3 m_velocity = { 0,0,0 };
public:

	// Setter and Getter functions
	float3* __host__ __device__ getPosition()
	{
		return & m_position;
	}
	void __host__ __device__ setPosition(float3& _position)
	{
		m_position = _position;
	}

	float3* __host__ __device__ getVelocity()
	{
		return &m_velocity;
	}
	void __host__ __device__ setVelocity(float3& _velocity)
	{
		m_position = _velocity;
	}

	// seeding particle
	void __host__ seedParticle(const float3& gridDimenstion)
	{
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.x);
		float y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.y);
		float z = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/gridDimenstion.z);
	}


	__device__ __host__ void move(const float& dt, VelocityField * p_velocityField)
	{
		this->updatePosition(dt);						//checked
		//this->updateVelocity(volume, p_velocityField);	//checked
	}


	__device__ __host__ void updatePosition(const float dt)
	{
		this->m_position.x += dt * (this->m_velocity.x);
		this->m_position.y += dt * (this->m_velocity.y);
		this->m_position.z += dt * (this->m_velocity.z);

	}

	__device__ __host__ void updateVelocity(const float* velocityField)
	{
		//int3 index = findIndex(diameter);

		//dimension of the grid data
		//int4 dim = { volume.getGridSize().x,volume.getGridSize().y,volume.getGridSize().z,volume.getFieldDim() };
		//uint mappedIndex_Vx = linearIndex(index.x, index.y, index.z, 0, dim);
		//float v_x = velocityField[mappedIndex_Vx];
		//float v_y = velocityField[mappedIndex_Vx + 1];
		//float v_z = velocityField[mappedIndex_Vx + 2];
		//float3 velocity = { v_x, v_y, v_z };
		//this->setVelocity(velocity);
	}
	__device__ __host__ int3 findIndex(float3  )
	{
		//float relative_x = (this->m_position.x) / (volume.getDiameter().x);
		//float relative_y = (this->m_position.y) / (volume.getDiameter().y);
		//float relative_z = (this->m_position.z) / (volume.getDiameter().z);

		int3 index = { 0,0,0 };
		//index.x = static_cast<int>(relative_x * (volume.getGridSize().x - 1));
		//index.y = static_cast<int>(relative_y * (volume.getGridSize().y - 1));
		//index.z = static_cast<int>(relative_z * (volume.getGridSize().z - 1));

		return index;
	}
};