#include "Particle.cuh"

// Explicit instantions
template class Particle<float>;
template class Particle<double>;

template <typename T>
__device__  void Particle<T>::updateVelocity(const float3& gridDiameter, const int3& gridSize, cudaTextureObject_t t_VelocityField)
{
	if (!outOfScope)
	{
		float3 linIndex = findRelative(gridDiameter);
		float4 velocity4D = tex3D<float4>(t_VelocityField, linIndex.x, linIndex.y, linIndex.z);
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
