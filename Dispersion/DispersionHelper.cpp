#include "DispersionHelper.h"


void seedParticle_ZY_Plane(Particle* particle, const float* gridDiameter, const int* gridSize, const float* seedDirection, const float & delta)
{
	// Size of the mesh in X, Y and Z direction
	float3 meshSize = { gridDiameter[0] / gridSize[0],gridDiameter[1] / gridSize[1], gridDiameter[2] / gridSize[2] };


		for (int y = 0; y < gridSize[1]; y++)
		{
			for (int z = 0; z < gridSize[2]; z++)
			{
				particle[x * gridSize[1] * gridSize[2] + y * gridSize[2] + z].m_position = { x * meshSize.x, y * meshSize.y, z * meshSize.z };

			}
		}
	}
}