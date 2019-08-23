#include "Particle.cuh"

// Explicit instantions
template class Particle<float>;
template class Particle<double>;

template <typename T>
__device__  void Particle<T>::updateVelocity(const float3& gridDiameter, const int3& gridSize, cudaTextureObject_t t_VelocityField)
{
	if (!outOfScope)
	{
		float3 relativePos = findRelative(gridDiameter);
		float4 velocity4D = tex3D<float4>(t_VelocityField, relativePos.z, relativePos.y, relativePos.x);
		float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
		this->setVelocity(velocity);
	}

}


template <typename T>
__device__ __host__ float3 Particle<T>::findRelative(const float3& gridDiameter)
{
	float3 relative_position = {
		(static_cast<float>(this->m_position.x)) / (gridDiameter.x),
		(static_cast<float>(this->m_position.y)) / (gridDiameter.y),
		(static_cast<float>(this->m_position.z)) / (gridDiameter.z)
	};
	return relative_position;
}

template <typename T>
__host__ void  Particle<T>::seedParticle(const float* gridDiameter, const float* seedBox, const float* seedBoxPos)
{
	this->m_position.x = +gridDiameter[0] /	2.0f - seedBox[0] / 2.0f + seedBoxPos[0] + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / seedBox[0]);
	this->m_position.y = +gridDiameter[1] / 2.0f - seedBox[1] / 2.0f + seedBoxPos[1] + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / seedBox[1]);
	this->m_position.z = +gridDiameter[2] / 2.0f - seedBox[2] / 2.0f + seedBoxPos[2] + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / seedBox[2]);

}

template <typename T>
__device__ void Particle<T>::checkPosition(const float3& gridDiameter)
{

	if (m_position.x >= gridDiameter.x)
	{
		m_position.x = gridDiameter.x;
		this->outOfScope = true;
	}
	else if (m_position.y >= gridDiameter.y)
	{
		m_position.y = gridDiameter.y;
		this->outOfScope = true;
	}
	else if (m_position.z >= gridDiameter.z)
	{
		m_position.z = gridDiameter.z;
		this->outOfScope = true;
	}

}

template <typename T>
__device__ void Particle<T>::updatePosition(const float dt)
{
	if (!outOfScope)
	{
		this->m_position.x += dt * (this->m_velocity.x);
		this->m_position.y += dt * (this->m_velocity.y);
		this->m_position.z += dt * (this->m_velocity.z);
	}

}

