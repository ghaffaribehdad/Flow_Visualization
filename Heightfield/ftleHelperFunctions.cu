#include "ftleHelperFunctions.h"
#include "../Cuda/helper_math.h"
#include "../Particle/Particle.h"
#include "../Cuda/CudaHelperFunctions.h"


enum RK4STEP
{
	EVEN = 0,
	ODD
};

__global__ void  traceDispersion3D_path_FTLE
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField_0,
	cudaTextureObject_t velocityField_1,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	RK4STEP RK4step,
	int timestep,
	unsigned int direction
) 
{
	// Extract dispersion options
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	int index = CUDA_INDEX;

	if (index < nParticles)
	{
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);



		// find the index of the particle (!!!!must be revised!!!!)
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 

		switch (RK4step)
		{
		case RK4STEP::EVEN:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_0, velocityField_1, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt);
			}
			break;

		case RK4STEP::ODD:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_1, velocityField_0, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt);
			}
			break;
		}

		float ftle = FTLE3D(&particle[index * FTLE_NEIGHBOR], dispersionOptions.initial_distance);

		if (dispersionOptions.timeNormalization)
			ftle /= (timestep*solverOptions.dt + 0.0000001f);
		float3 position = particle[index * FTLE_NEIGHBOR].m_position;


		// initialize texels
		float4 heightTexel = { 0.0f,0.0,0.0,0.0 };
		float4 extraTexel = { 0.0f, 0.0f ,0.0f, 0.0f };


		switch (direction)
		{
		case FTLE_Direction::FORWARD_FTLE:
			extraTexel.x = ftle;
			heightTexel.x = position.y;
			// copy it in the surface3D
			surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, timestep);
			surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, timestep);

			break;

		case FTLE_Direction::BACKWARD_FTLE:
			extraTexel = ValueAtXYZ_Surface_float4(heightFieldSurface3D_extra, make_int3(index_x, index_y, timestep));
			extraTexel.y = ftle;
			surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, timestep);

			break;
		}


	}
}


__global__ void  traceDispersion3D_path_FSLE
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField_0,
	cudaTextureObject_t velocityField_1,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	FSLEOptions fsleOptions,
	RK4STEP RK4step,
	int timestep
)
{
	// Extract dispersion options
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	int index = CUDA_INDEX;

	if (index < nParticles)
	{
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);



		// find the index of the particle (!!!!must be revised!!!!)
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 

		switch (RK4step)
		{
		case RK4STEP::EVEN:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_0, velocityField_1, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt);
			}
			break;

		case RK4STEP::ODD:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_1, velocityField_0, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt);
			}
			break;
		}

		float fsle = 0;

		if (particle[index * FTLE_NEIGHBOR].diverged)
		{
			fsle = particle[index * FTLE_NEIGHBOR].fsle;
		}
		else
		{
			float averageDist = averageNeighborDistance(&particle[index * FTLE_NEIGHBOR]);
			if (averageDist > fsleOptions.separation_factor * dispersionOptions.initial_distance)
			{
				particle[index * FTLE_NEIGHBOR].diverged = true;
				fsle = log(fsleOptions.separation_factor) / (timestep * solverOptions.dt);
				particle[index * FTLE_NEIGHBOR].fsle = fsle;
			}
		}
		

		// extract the height
		float3 position = particle[index * FTLE_NEIGHBOR].m_position;

		float4 heightTexel = { position.y,0.0,0.0,0.0 };
		float4 extraTexel = { fsle, 0.0f ,0.0f, 0.0f };


		// copy it in the surface3D
		surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, timestep);
		surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, timestep);

	}
}


__device__ float averageNeighborDistance(Particle* particles)
{
	float distance =
		magnitude(particles[0].m_position - particles[1].m_position) +
		magnitude(particles[0].m_position - particles[2].m_position) +
		magnitude(particles[0].m_position - particles[3].m_position) +
		magnitude(particles[0].m_position - particles[4].m_position) +
		magnitude(particles[0].m_position - particles[5].m_position) +
		magnitude(particles[0].m_position - particles[6].m_position);
	return distance / 6.0f;
}

__device__ float FTLE3D(Particle* particles,const float & distance)
{
	dMat3X3 d_Flowmap(0.0f,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
	
	// Calculate the Jacobian of the Flow Map
	d_Flowmap.r1.x = 0.5 * (particles[1].m_position.x - particles[2].m_position.x) / distance;
	d_Flowmap.r2.x = 0.5 * (particles[3].m_position.x - particles[4].m_position.x) / distance;
	d_Flowmap.r3.x = 0.5 * (particles[5].m_position.x - particles[6].m_position.x) / distance;

	d_Flowmap.r1.y = 0.5 * (particles[1].m_position.y - particles[2].m_position.y) / distance;
	d_Flowmap.r2.y = 0.5 * (particles[3].m_position.y - particles[4].m_position.y) / distance;
	d_Flowmap.r3.y = 0.5 * (particles[5].m_position.y - particles[6].m_position.y) / distance;

	d_Flowmap.r1.z = 0.5 * (particles[1].m_position.z - particles[2].m_position.z) / distance;
	d_Flowmap.r2.z = 0.5 * (particles[3].m_position.z - particles[4].m_position.z) / distance;
	d_Flowmap.r3.z = 0.5 * (particles[5].m_position.z - particles[6].m_position.z) / distance;

	// Find the Delta Tensor
	dMat3X3 td_Flowmap = transpose(d_Flowmap);
	dMat3X3 delta = mult(d_Flowmap, td_Flowmap);
	//float3 eigen = { 0.0f,0.0f,0.0f };
	
	// Calculate and sort the eigenvalues
	//eigensolveHasan(delta, eigen);

	float lambda_max = eigenValueMax(delta);


	//return (1.0f/T) * logf(sqrtf(lambda_max)); // Time dependent
	return logf(sqrtf(lambda_max)); // Time dependent

}


__device__ float FTLE3D_test(Particle* particles, const float & distance)
{
	float averageDist = 0;
	for (int i = 1; i < 7; i++)
	{
		averageDist += magnitude(particles[0].m_position - particles[i].m_position);
	}
	return averageDist;
}