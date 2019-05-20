#pragma once

#include "cuda_runtime.h"
class VelocityField	
{


public:

	// constructor
	__device__ __host__	VelocityField(float3 _diamter,
		int3 _gridSize,
		bool _PBC
	);

	__device__ __host__ VelocityField();

	__device__ __host__ bool getPBC() const;
	__device__ __host__ float* getVelocityField();

	// setter
	__device__ __host__ void setGridSize(const int3& gridSize);
	__device__ __host__ void setGridDiameter(const float3& gridDiameter);
	__device__ __host__ void setVelocityField(float* _velocityField);

	
private:

	// VelocityField
	float* m_velcocityField = nullptr;

	// return the grid point correspond to the position
	__device__ __host__ int3 extractGridPosition(float3 position) const;

	// size of the mesh (x,y,z)
	int3 m_gridSize;

	// diameters of the volume 
	float3 m_gridDiameter;

	// priodic boundary condition; true if the boundary conditions are periodic
	bool m_PBC = false;

};