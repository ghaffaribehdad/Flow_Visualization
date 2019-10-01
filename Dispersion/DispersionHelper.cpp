#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"


// Seed 2 particles in each ZY-Plane grid points
// takes arrayes of three floats for gridDiameter and gridSize
// takes arrays of particle of size of 2 * gridSize[1]*gridSize[2]
// X_Slice: the x coordinate bisection
// Seed Direction: Vector connecting two neighbouring particles ( wall-normal direction is recommended)
// Delta is the distance between the two neighbouring particles
void seedParticle_ZY_Plane(Particle* particle, const float* gridDiameter, const int* gridSize, const int & x_slice, const float3 & seedDirection, const float & delta)
{
	// Size of the mesh in X, Y and Z direction
	float3 meshSize = {gridDiameter[1] / gridSize[1], gridDiameter[2] / gridSize[2] };


		for (int y = 0; y < gridSize[1]; y++)
		{
			for (int z = 0; z < 2*gridSize[2]; z+=2)
			{
				float3 center = { x_slice * meshSize.x, y * meshSize.y, z * meshSize.z };


				particle[y * gridSize[2] + z].m_position		=  center + (delta / 2.0f) * seedDirection;
				particle[y * gridSize[2] + z + 1].m_position	=  center - (delta / 2.0f) * seedDirection;

			}
		}
	
}