#include "ParticleHelperFunctions.h"
#include <math.h>
#include <random>


void seedParticle_tiltedPlane(Particle* particle, const float3& gridDiameter, const int2& gridSize, const float& y_slice, const float& tilt)
{
	// Size of the mesh in X, Y and Z direction
	float meshSize_x = (float)gridDiameter.x / (gridSize.x - 1);
	float meshSize_z = (float)gridDiameter.z / (gridSize.y - 1);
	float height_step = gridDiameter.y * sinf(tilt * 3.1415f / 180.0f) / gridSize.x;

	for (int x = 0; x < gridSize.x; x++)
	{
		for (int z = 0; z < gridSize.y; z++)
		{

			particle[x * gridSize.y + z].m_position = { meshSize_x * x,y_slice + x * height_step,meshSize_z * z };

		}
	}

}

void seedParticle_ZY_Plane(Particle* particle, float* gridDiameter, const int* gridSize, const float& y_slice)
{
	// Size of the mesh in X, Y and Z direction
	float meshSize_x = (float)gridDiameter[0] / (gridSize[0] - 1);
	float meshSize_z = (float)gridDiameter[2] / (gridSize[1] - 1);


	for (int x = 0; x < gridSize[0]; x++)
	{
		for (int z = 0; z < gridSize[1]; z++)
		{

			particle[x * gridSize[0] + z].m_position = { meshSize_x * x,y_slice,meshSize_z * z };

		}
	}

}



void seedParticle_ZY_Plane_FTLE
(
	Particle* particle,
	const float3& gridDiameter,
	const int2&	gridSize,
	const float& y_slice,
	const float& tilt,
	const float& delta_x
)
{
	// Size of the mesh in X, Y and Z direction
	float meshSize_x = (float)gridDiameter.x / (gridSize.x - 1);
	float meshSize_z = (float)gridDiameter.z / (gridSize.y - 1);
	float height_step = gridDiameter.y * sinf(tilt * 3.1415f / 180.0f) / gridSize.x;

	for (int x = 0; x < gridSize.x; x++)
	{
		for (int z = 0; z < gridSize.y; z++)
		{

			particle[x * gridSize.y * 7 + z * 7 + 0].m_position = { meshSize_x * x,y_slice + z * height_step,meshSize_z * z };				// (x , y , z)
			particle[x * gridSize.y * 7 + z * 7 + 1].m_position = { meshSize_x * x + delta_x ,y_slice + z * height_step,meshSize_z * z };	// (x+1, y ,z)
			particle[x * gridSize.y * 7 + z * 7 + 2].m_position = { meshSize_x * x - delta_x ,y_slice + z * height_step,meshSize_z * z };	// (x-1, y ,z)
			particle[x * gridSize.y * 7 + z * 7 + 3].m_position = { meshSize_x * x,y_slice + z * height_step + delta_x,meshSize_z * z };	// (x, y+1 ,z)
			particle[x * gridSize.y * 7 + z * 7 + 4].m_position = { meshSize_x * x,y_slice + z * height_step - delta_x,meshSize_z * z };	// (x, y-1 ,z)
			particle[x * gridSize.y * 7 + z * 7 + 5].m_position = { meshSize_x * x,y_slice + z * height_step,meshSize_z * z + delta_x };	// (x, y, z+1)
			particle[x * gridSize.y * 7 + z * 7 + 6].m_position = { meshSize_x * x,y_slice + z * height_step,meshSize_z * z - delta_x };	// (x, y, z-1)

		}
	}

}


 void seedParticleGridPoints(Particle* particle, const SolverOptions* solverOptions)
{
	float3 gridMeshSize =
	{
		solverOptions->seedBox[0] / (float)solverOptions->seedGrid[0],
		solverOptions->seedBox[1] / (float)solverOptions->seedGrid[1],
		solverOptions->seedBox[2] / (float)solverOptions->seedGrid[2],
	};

	for (int x = 0; x < solverOptions->seedGrid[0]; x++)
	{
		for (int y = 0; y < solverOptions->seedGrid[1]; y++)
		{
			for (int z = 0; z < solverOptions->seedGrid[2]; z++)
			{
				int index = x * solverOptions->seedGrid[1] * solverOptions->seedGrid[2] + y * solverOptions->seedGrid[2] + z;
				particle[index].m_position =
				{

					solverOptions->gridDiameter[0] / 2.0f -
					solverOptions->seedBox[0] / 2.0f +
					solverOptions->seedBoxPos[0] + (float)x * gridMeshSize.x,

					solverOptions->gridDiameter[1] / 2.0f -
					solverOptions->seedBox[1] / 2.0f +
					solverOptions->seedBoxPos[1] + (float)y * gridMeshSize.y,

					solverOptions->gridDiameter[2] / 2.0f -
					solverOptions->seedBox[2] / 2.0f +
					solverOptions->seedBoxPos[2] + (float)z * gridMeshSize.z
				};

			}
		}
	}

}


__host__ void seedParticleRandom(Particle* particle, const SolverOptions* solverOptions)
{

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	generator.seed(solverOptions->counter);

	for (int i = 0; i < solverOptions->lines_count; i++)
	{
		particle[i].m_position.x = solverOptions->gridDiameter[0] / 2.0f -
			solverOptions->seedBox[0] / 2.0f + solverOptions->seedBoxPos[0] +
			(float)distribution(generator) * solverOptions->seedBox[0];

		distribution.reset();

		particle[i].m_position.y = solverOptions->gridDiameter[1] / 2.0f -
			solverOptions->seedBox[1] / 2.0f + solverOptions->seedBoxPos[1] +
			(float)distribution(generator) * solverOptions->seedBox[1];

		distribution.reset();

		particle[i].m_position.z = solverOptions->gridDiameter[2] / 2.0f -
			solverOptions->seedBox[2] / 2.0f + solverOptions->seedBoxPos[2] +
			(float)distribution(generator) * solverOptions->seedBox[2];
		distribution.reset();

	}


}