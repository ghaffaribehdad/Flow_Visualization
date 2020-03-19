#include "ftleHelperFunctions.h"
#include "../Cuda/helper_math.h"
#include "../Particle/Particle.h"
#include "../Cuda/CudaHelperFunctions.h"

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
	int timestep
) 
{
	// Extract dispersion options
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	int index = CUDA_INDEX;

	if (index < nParticles)
	{
		float3 gridDiameter = ARRAYTOFLOAT3(solverOptions.gridDiameter);
		int3 gridSize = ARRAYTOINT3(solverOptions.gridSize);



		// find the index of the particle (!!!!must be revised!!!!)
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 

		switch (RK4step)
		{
		case RK4STEP::EVEN:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_0, velocityField_1, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt, true);
			}
			break;

		case RK4STEP::ODD:
			for (int i = 0; i < FTLE_NEIGHBOR; i++)
			{
				RK4Path(velocityField_1, velocityField_0, &particle[index * FTLE_NEIGHBOR + i], gridDiameter, gridSize, dispersionOptions.dt, true);
			}
			break;
		}

		float ftle = FTLE3D(&particle[index * FTLE_NEIGHBOR], dispersionOptions.ftleDistance, dispersionOptions.dt * (timestep+1));

		// extract the height
		float3 position = particle[index * FTLE_NEIGHBOR].m_position;
		float3 velocity = particle[index * FTLE_NEIGHBOR].m_velocity;

		float4 heightTexel = { position.y,0.0,0.0,0.0 };
		float4 extraTexel = { ftle, 0.0f ,0.0f, 0.0f};

		
		// copy it in the surface3D
		surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, timestep);
		surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, timestep);

	}
}



__device__ float FTLE3D(Particle* particles, float distance, float T)
{
	fMat3X3 d_Flowmap(0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f);
	
	// Calculate the Jacobian of the Flow Map
	d_Flowmap.r1.x = 0.5f * (particles[1].m_position.x - particles[2].m_position.x) / distance;
	d_Flowmap.r1.y = 0.5f * (particles[3].m_position.x - particles[4].m_position.x) / distance;
	d_Flowmap.r1.z = 0.5f * (particles[5].m_position.x - particles[6].m_position.x) / distance;

	d_Flowmap.r2.x = 0.5f * (particles[1].m_position.y - particles[2].m_position.y) / distance;
	d_Flowmap.r2.y = 0.5f * (particles[3].m_position.y - particles[4].m_position.y) / distance;
	d_Flowmap.r2.z = 0.5f * (particles[5].m_position.y - particles[6].m_position.y) / distance;

	d_Flowmap.r3.x = 0.5f * (particles[1].m_position.z - particles[2].m_position.z) / distance;
	d_Flowmap.r3.y = 0.5f * (particles[3].m_position.z - particles[4].m_position.z) / distance;
	d_Flowmap.r3.z = 0.5f * (particles[5].m_position.z - particles[6].m_position.z) / distance;

	// Find the Delta Tensor
	fMat3X3 td_Flowmap = transpose(d_Flowmap);
	fMat3X3 delta = mult(td_Flowmap, d_Flowmap);
	float3 eigen = { 0.0f,0.0f,0.0f };
	
	// Calculate and sort the eigenvalues
	eigensolveHasan(delta, eigen);

	float lambda_max = eigen.z;


	//return (1.0f/T) * logf(sqrtf(lambda_max)); // Time dependent
	return logf(sqrtf(lambda_max));
}