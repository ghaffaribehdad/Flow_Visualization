#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"


template __global__ void hieghtfieldGradient<struct IsosurfaceHelper::Position>(cudaSurfaceObject_t heightFieldSurface, cudaSurfaceObject_t heightFieldSurface_gradient ,DispersionOptions dispersionOptions, SolverOptions	solverOptions);


// Seed particles in each ZY-Plane grid points
// takes arrayes of three floats for gridDiameter and 2 integer gridSize

void seedParticle_ZY_Plane(Particle* particle, float* gridDiameter, const int* gridSize, const float & y_slice)
{
	// Size of the mesh in X, Y and Z direction
	float meshSize_x = (float)gridDiameter[0] / ( gridSize[0]-1);
	float meshSize_z = (float)gridDiameter[2] / ( gridSize[1]-1);


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
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions	
)
{
	// Extract dispersion options
	float dt = dispersionOptions.dt;
	int timeStep = dispersionOptions.timeStep;
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];


	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;
	 
	if (index < nParticles)
	{
		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2],
		};

		// find the index of the particle
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);



		// Trace particle using RK4 
		for (int i = 0; i < timeStep; i++)
		{

			RK4Stream(velocityField, &particle[index], gridDiameter, dt);
		}
		

		float height = particle[index].m_position.y;
	
		surf2Dwrite(height, heightFieldSurface,  sizeof(float) *index_x, index_y);
	}
}



__global__ void traceDispersion3D
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
)
{
	// Extract dispersion options
	float dt = dispersionOptions.dt;
	int timeStep = dispersionOptions.timeStep;
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];


	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < nParticles)
	{
		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2],
		};

		// find the index of the particle
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 
		for (int time = 0; time < timeStep; time++)
		{

			// Advect the particle
			RK4Stream(velocityField, &particle[index], gridDiameter, dt);
		
			// extract the height
			float4 height = { particle[index].m_position.y,0.0,0.0,0.0 };

			// copy it in the surface3D
			surf3Dwrite(height, heightFieldSurface3D, sizeof(float4) * index_x, index_y,time);

		}
	}
}



template <typename Observable>
__global__ void hieghtfieldGradient
(
	cudaSurfaceObject_t heightFieldSurface,
	cudaSurfaceObject_t heightFieldSurface_gradient,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{

	Observable observable;
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	if (index < gridPoints)
	{



		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);
		
		float2 gradient = { 0.0f,0.0f };

		// check the boundaries
		if (index_x % (dispersionOptions.gridSize_2D[0] - 1) == 0.0f || index_y % (dispersionOptions.gridSize_2D[1] - 1) == 0.0f)
		{
			// do nothing

		}
		else
		{
			gradient = observable.GradientAtXY_Grid(heightFieldSurface, make_int2(index_x, index_y));
		}


		

		float4 texel = { 0,0,0,0 };
		texel.x = observable.ValueAtXY_Surface_float(heightFieldSurface, make_int2(index_x, index_y));
		texel.y = gradient.x;
		texel.z = gradient.y;

		surf2Dwrite(texel, heightFieldSurface_gradient, 4 * sizeof(float) * index_x, index_y);
		
	}
}



template <typename Observable>
__global__ void hieghtfieldGradient3D
(
	cudaSurfaceObject_t heightFieldSurface3D,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{

	Observable observable;
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	if (index < gridPoints)
	{


		for (int time = 0; time < dispersionOptions.timeStep; time++)
		{
			int index_y = index / dispersionOptions.gridSize_2D[1];
			int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);

			float2 gradient = { 0.0f,0.0f };

			// check the boundaries
			if (index_x % (dispersionOptions.gridSize_2D[0] - 1) == 0.0f || index_y % (dispersionOptions.gridSize_2D[1] - 1) == 0.0f)
			{
				// do nothing

			}
			else
			{
				gradient = observable.GradientAtXYZ_Grid(heightFieldSurface3D, make_int3(index_x, index_y,time));
			}




			float4 texel = { 0,0,0,0 };
			texel.x = observable.ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y,time)).x;
			texel.y = gradient.x;
			texel.z = gradient.y;

			surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * index_x, index_y,time);

		}


	}
}