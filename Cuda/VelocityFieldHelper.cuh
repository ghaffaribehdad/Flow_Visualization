#pragma once

//#include "cuda_runtime.h"
//
//__device__  int3 extractGridPosition(float3 position, int3 gridSize) const
//{
//	int x_grid = static_cast<int> (m_gridSize.x * (position.x / m_gridDiameter.x));
//
//	int y_grid = static_cast<int> (m_gridSize.y * (position.y / m_gridDiameter.y));
//
//	int z_grid = static_cast<int> (m_gridSize.z * (position.z / m_gridDiameter.z));
//
//	int3 gridposition = { x_grid, y_grid, z_grid };
//
//	return gridposition;
//}