
#include "VelocityField.h"
__device__ __host__ VelocityField::VelocityField()
{
	this->m_gridSize = { 0,0,0 };
	this->m_gridDiameter = { 0,0,0 };
	this->m_PBC = true;
}

__device__ __host__ VelocityField::VelocityField(float3 _diamter, int3 _gridSize, bool _PBC)
{
	this->m_gridSize = _gridSize;
	this->m_gridDiameter = _diamter;
	this->m_PBC = _PBC;
};

__device__ __host__ bool VelocityField::getPBC() const
{
	return this->m_PBC;
}

__device__ __host__ float* VelocityField::getVelocityField()
{
	return m_velcocityField;
}



__device__ __host__ int3 VelocityField::extractGridPosition(float3 position) const
{
	int x_grid = static_cast<int> (m_gridSize.x * (position.x / m_gridDiameter.x));

	int y_grid = static_cast<int> (m_gridSize.y * (position.y / m_gridDiameter.y));

	int z_grid = static_cast<int> (m_gridSize.z * (position.z / m_gridDiameter.z));

	int3 gridposition = { x_grid, y_grid, z_grid };

	return gridposition;
};



// a pointer from shared memory would be passed to the setter

__device__ __host__ void VelocityField::setGridSize(const int3 & gridSize)
{
	this->m_gridSize = gridSize;
}
__device__ __host__ void VelocityField::setGridDiameter(const float3 & gridDiameter)
{
	this->m_gridDiameter = gridDiameter;
}

__device__ __host__ void VelocityField::setVelocityField(float* _velocityField)
{
	this->m_velcocityField = _velocityField;
}