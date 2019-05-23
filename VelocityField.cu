
#include "VelocityField.cuh"

template class VelocityField<float>;
template class VelocityField<double>;

template <typename T>
__device__ __host__ VelocityField<T>::VelocityField()
{
	this->m_gridSize = { 0,0,0 };
	this->m_gridDiameter = { 0,0,0 };
	this->m_PBC = true;
}

template <typename T>
__device__ __host__ VelocityField<T>::VelocityField(float3 _diamter, int3 _gridSize, bool _PBC)
{
	this->m_gridSize = _gridSize;
	this->m_gridDiameter = _diamter;
	this->m_PBC = _PBC;
};

template <typename T>
__device__ __host__ bool VelocityField<T>::getPBC() const
{
	return this->m_PBC;
}

template <typename T>
__device__ __host__ T* VelocityField<T>::getVelocityField()
{
	return m_velocityField;
}


template <typename T>
__device__ __host__ int3 VelocityField<T>::extractGridPosition(float3 position) const
{
	int x_grid = static_cast<int> (m_gridSize.x * (position.x / m_gridDiameter.x));

	int y_grid = static_cast<int> (m_gridSize.y * (position.y / m_gridDiameter.y));

	int z_grid = static_cast<int> (m_gridSize.z * (position.z / m_gridDiameter.z));

	int3 gridposition = { x_grid, y_grid, z_grid };

	return gridposition;
};



// a pointer from shared memory would be passed to the setter
template <typename T>
__device__ __host__ void VelocityField<T>::setGridSize(const int3 & gridSize)
{
	this->m_gridSize = gridSize;
}

template <typename T>
__device__ __host__ void VelocityField<T>::setGridDiameter(const float3 & gridDiameter)
{
	this->m_gridDiameter = gridDiameter;
}

template <typename T>
__device__ __host__ void VelocityField<T>::setVelocityField(T* _velocityField)
{
	this->m_velocityField = _velocityField;
}