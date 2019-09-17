#include "Particle.h"



__device__  void Particle::updateVelocity(const float3& gridDiameter, cudaTextureObject_t t_VelocityField)
{
	if (!outOfScope)
	{
		float3 relativePos = findRelative(gridDiameter);
		float4 velocity4D = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z);
		float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
		this->setVelocity(velocity);
	}

}


__device__ __host__ float3 Particle::findRelative(const float3& gridDiameter)
{
	float3 relative_position = {
		(static_cast<float>(this->m_position.x)) / (gridDiameter.x),
		(static_cast<float>(this->m_position.y)) / (gridDiameter.y),
		(static_cast<float>(this->m_position.z)) / (gridDiameter.z)
	};
	return relative_position;
}



__device__ void Particle::checkPosition(const float3& gridDiameter)
{

	if (m_position.x >= gridDiameter.x || m_position.x <= 0)
	{
		m_position.x = gridDiameter.x;
		this->outOfScope = true;
	}
	else if (m_position.y >= gridDiameter.y || m_position.y<= 0)
	{
		m_position.y = gridDiameter.y;
		this->outOfScope = true;
	}
	else if (m_position.z >= gridDiameter.z || m_position.z <= 0)
	{
		m_position.z = gridDiameter.z;
		this->outOfScope = true;
	}

}


__device__ void Particle::updatePosition(const float dt)
{
	if (!outOfScope)
	{
		this->m_position.x += dt * (this->m_velocity.x);
		this->m_position.y += dt * (this->m_velocity.y);
		this->m_position.z += dt * (this->m_velocity.z);
	}

}

