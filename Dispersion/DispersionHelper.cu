#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"


// Seed particles in each ZY-Plane grid points
// takes arrayes of three floats for gridDiameter and 2 integer gridSize

void seedParticle_ZY_Plane(Particle* particle, const float* gridDiameter, const int* gridSize, const float & y_slice)
{
	// Size of the mesh in X, Y and Z direction
	float meshSize_x = gridDiameter[0] / ((float) gridSize[0]-1);
	float meshSize_z = gridDiameter[3] / ((float) gridSize[1]-1);


		for (int x = 0; x < gridSize[0]; x++)
		{
			for (int z = 0; z < gridSize[1]; z++)
			{
				
				particle[x * gridSize[0] + z].m_position = { meshSize_x * x,y_slice,meshSize_z * z };

			}
		}
	
}