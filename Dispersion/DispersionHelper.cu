#include "DispersionHelper.h"
#include "../Cuda/helper_math.h"
#include "../Cuda/CudaHelperFunctions.h"

//Explicit Instantiation
template __global__ void heightFieldGradient3D<struct FetchTextureSurface::Channel_X>\
(\
	cudaSurfaceObject_t heightFieldSurface,\
	DispersionOptions dispersionOptions,\
	SolverOptions	solverOptions\
);

template __global__ void fluctuationfieldGradient3D<struct FetchTextureSurface::Channel_X>\
(\
	cudaSurfaceObject_t heightFieldSurface3D,
	SolverOptions solverOptions,
	FluctuationheightfieldOptions fluctuationOptions
);




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
	int timeStep = dispersionOptions.timestep;
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];


	int index = CUDA_INDEX;
	 
	if (index < nParticles)
	{

		// find the index of the particle
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);



		// Trace particle using RK4 
		for (int i = 0; i < timeStep; i++)
		{

			RK4Stream(velocityField, &particle[index], Array2Float3(solverOptions.gridDiameter), Array2Int3(solverOptions.gridSize), dt);
		}
		

		float height = particle[index].m_position.y;
	
		surf2Dwrite(height, heightFieldSurface,  sizeof(float) *index_x, index_y);
	}
}




__global__ void  traceDispersion3D_path
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


		// find the index of the particle
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 

		switch (RK4step)
		{
		case EVEN: // => EVEN

			RK4Path(velocityField_0, velocityField_1, &particle[index], gridDiameter, gridSize, dispersionOptions.dt,true);

			break;

		case ODD: // => ODD

			RK4Path(velocityField_1, velocityField_0, &particle[index], gridDiameter, gridSize, dispersionOptions.dt,true);

			break;
		}
	

			// extract the height
			float3 position = particle[index].m_position;
			float3 velocity = particle[index].m_velocity;

			float4 heightTexel = { position.y,0.0,0.0,position.x };
			float4 extraTexel = { position.z, velocity.x ,velocity.x, velocity.z };


			// copy it in the surface3D
			surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_y, index_x, timestep);
			surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_y, index_x, timestep);

	}
}


__global__ void  traceDispersion3D
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
	int traceTime = solverOptions.lastIdx-solverOptions.firstIdx;
	int nParticles = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];


	int index = CUDA_INDEX;

	if (index < nParticles)
	{

		// find the index of the particle
		int index_y = index / dispersionOptions.gridSize_2D[1];
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);


		// Trace particle using RK4 
		for (int time = 0; time < traceTime; time++)
		{

			// Advect the particle
			RK4Stream(velocityField, &particle[index], Array2Float3(solverOptions.gridDiameter),Array2Int3(solverOptions.gridSize), dt);
		
			// extract the height
			float4 height = { particle[index].m_position.y,0.0,0.0,0.0 };

			// copy it in the surface3D
			surf3Dwrite(height, heightFieldSurface3D, sizeof(float4) * index_x, index_y,time);

		}
	}
}







template <typename Observable>
__global__ void heightFieldGradient3D
(
	cudaSurfaceObject_t heightFieldSurface3D,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{

	
	int index = CUDA_INDEX;

	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	if (index < gridPoints)
	{
		Observable observable;
		for (int time = 0; time < solverOptions.lastIdx - solverOptions.firstIdx; time++)
		{
			int index_y = index / dispersionOptions.gridSize_2D[1];
			int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);

			int3 gridSize = { dispersionOptions.gridSize_2D[0] , dispersionOptions.gridSize_2D[1], solverOptions.lastIdx - solverOptions.firstIdx };

			float3 gradient = { 0.0f,0.0f,0.0f };


			gradient = observable.GradientAtXYZ_Surf(heightFieldSurface3D, make_int3(index_x, index_y,time),ARRAYTOFLOAT3(solverOptions.gridDiameter),gridSize);
			gradient = { normalize(make_float3(1.0,gradient.x, gradient.y)).y, normalize(make_float3(1.0,gradient.x, gradient.y)).z };
			

			float4 texel = { 0,0,0,0 };
			texel.x = ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y,time)).x;
			texel.y = gradient.x;
			texel.z = gradient.y;
			texel.w = ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y, time)).w;

			surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * index_x, index_y,time);

		}


	}
}



__global__ void heightFieldGradient3DFTLE
(
	cudaSurfaceObject_t heightFieldSurface3D,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{
	int index = CUDA_INDEX;

	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];
	

	if (index < gridPoints)
	{
		int3 gridSize = { dispersionOptions.gridSize_2D[0] , dispersionOptions.gridSize_2D[1], solverOptions.lastIdx - solverOptions.firstIdx - 1 };
		FetchTextureSurface::Channel_X fetchSurface;
		for (int time = 1; time < solverOptions.lastIdx - solverOptions.firstIdx-1; time++)
		{
			int index_y = index / dispersionOptions.gridSize_2D[1];
			int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);

			float3 gradient = fetchSurface.GradientAtXYZ_Surf
			(
				heightFieldSurface3D, make_int3(index_x, index_y, time),
				ARRAYTOFLOAT3(solverOptions.gridDiameter),
				gridSize
			);
			
			gradient = normalize(gradient);
		

			float4 texel = { 0,0,0,0 };
			texel.x = ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y, time)).x;
			texel.y = gradient.x;
			texel.z = gradient.y;
			texel.w = gradient.z;

			surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, time);

		}


	}
}



template <typename Observable>
__global__ void fluctuationfieldGradient3D
(
	cudaSurfaceObject_t heightFieldSurface3D,
	SolverOptions solverOptions,
	FluctuationheightfieldOptions fluctuationOptions
)
{

	Observable observable;
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;
	
	int timeDim = 1 + solverOptions.lastIdx - solverOptions.firstIdx;

	int3 gridSize = { solverOptions.gridSize[2],static_cast<int>(fluctuationOptions.wallNormalgridSize), timeDim };

	// index in this case it the wall-normal position ( second index)
	if (index < gridSize.y)
	{
		// is the first index 
		for (int z = 0; z < solverOptions.gridDiameter[2] ; z++)
		{
			// t is the third index
			for (int t = 0; t < timeDim; t++)
			{
				float3 gradient = { 0.0f,0.0f,0.0f };


				gradient = observable.GradientAtXYZ_Surf(heightFieldSurface3D, make_int3(z,index, t), ARRAYTOFLOAT3(solverOptions.gridDiameter), gridSize);
				float3 gradient3D = normalize(make_float3(1.0, gradient.x, gradient.z));

				float4 texel = ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(z, index, t));

				texel.z = gradient3D.y;
				texel.w = gradient3D.z;

				surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * z, index, t);


			}

		}


	}
}
