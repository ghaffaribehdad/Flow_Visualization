#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"

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


__global__ void traceDispersion
(
	int timeStep,
	float dt,
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface,
	cudaTextureObject_t velocityField,
	int nParticles,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions	
)
{
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < nParticles)
	{
		float3 gridDiamter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2],
		};

		int index_x = index / dispersionOptions.gridSize_2D[1];
		int index_y = index - (index_x * dispersionOptions.gridSize_2D[1]);

		for (int i = 0; i < timeStep; i++)
		{
			float4 rgba = { particle[index].m_position.x, particle[index].m_position.y,particle[index].m_position.z, 0.0f};

			surf3Dwrite(rgba, heightFieldSurface, 4 * sizeof(float) * index_x, index_y,i);

			RK4Stream(velocityField, &particle[index], gridDiamter, dt);

		}
	}
}