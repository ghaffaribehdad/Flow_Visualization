#pragma once

#include "cuda_runtime.h"
#include <vector>

template <class T>
class VelocityField	
{


public:


	__device__ __host__ VelocityField();

	__device__ __host__ T* getVelocityField();

	// setter
	__device__ __host__ void setGridSize(const int3& gridSize);
	__device__ __host__ void setGridDiameter(const float3& gridDiameter);
	__device__ __host__ void setVelocityField(T* _velocityField);

	// getter
	__device__ __host__ float3 getGridDiameter()
	{
		return m_gridDiameter;
	}
	__device__ __host__ int3 getGridSize()
	{
		return m_gridSize;
	}

private:
	__host__ std::vector<char>* readVelocityField(unsigned int id = 0);

	// VelocityField
	T* m_velocityField = nullptr;


	// return the grid point correspond to the position
	__device__ __host__ int3 extractGridPosition(float3 position) const;

	// size of the mesh (x,y,z)
	int3 m_gridSize;

	// diameters of the volume 
	float3 m_gridDiameter;

	// priodic boundary condition; true if the boundary conditions are periodic
	bool m_PBC = false;

};