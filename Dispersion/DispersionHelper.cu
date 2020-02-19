#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//Cuda/CudaHelperFunctions.h"

//Explicit Instantiation
template __global__ void heightFieldGradient<struct IsosurfaceHelper::Position>(cudaSurfaceObject_t heightFieldSurface,\
	cudaSurfaceObject_t heightFieldSurface_gradient ,\
	DispersionOptions dispersionOptions,\
	SolverOptions	solverOptions
);
template __global__ void heightFieldGradient3D<struct IsosurfaceHelper::Position>\
(\
	cudaSurfaceObject_t heightFieldSurface,\
	DispersionOptions dispersionOptions,\
	SolverOptions	solverOptions\
);

template __global__ void fluctuationfieldGradient3D<struct IsosurfaceHelper::Position>\
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

		switch (RK4step)
		{
		case 0: // => EVEN

			RK4Path(velocityField_0, velocityField_1, &particle[index], gridDiameter, dispersionOptions.dt,true);

			break;

		case 1: // => ODD

			RK4Path(velocityField_1, velocityField_0, &particle[index], gridDiameter, dispersionOptions.dt,true);

			break;
		}
	

			// extract the height
			float3 position = particle[index].m_position;
			float3 velocity = particle[index].m_velocity;

			float4 heightTexel = { position.y,0.0,0.0,position.x };
			float4 extraTexel = { position.z, velocity.x ,velocity.x, velocity.z };


			// copy it in the surface3D
			surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, timestep);
			surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, timestep);

	}
}


__global__ void  traceDispersion3D_extra
(
	Particle* particle,
	cudaSurfaceObject_t heightFieldSurface3D,
	cudaSurfaceObject_t heightFieldSurface3D_extra,
	cudaTextureObject_t velocityField,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions
)
{
	// Extract dispersion options
	float dt = dispersionOptions.dt;
	int tracingTime = solverOptions.lastIdx - solverOptions.firstIdx;
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
		for (int time = 0; time < tracingTime; time++)
		{
			for(int i = 0 ; i < dispersionOptions.sampling_step; i++)
			{ 
				// Advect the particle
				RK4Stream(velocityField, &particle[index], gridDiameter, dt);
			}
			// extract the height
			float3 position = particle[index].m_position;
			float3 velocity = particle[index].m_velocity;

			float4 heightTexel = { position.y,0.0,0.0,position.x };
			float4 extraTexel = { position.z, velocity.x ,velocity.x, velocity.z };


			// copy it in the surface3D
			surf3Dwrite(heightTexel, heightFieldSurface3D, sizeof(float4) * index_x, index_y, time);
			surf3Dwrite(extraTexel, heightFieldSurface3D_extra, sizeof(float4) * index_x, index_y, time);


		}
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
		for (int time = 0; time < traceTime; time++)
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
__global__ void heightFieldGradient
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
__global__ void heightFieldGradient3D
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


		for (int time = 0; time < solverOptions.lastIdx - solverOptions.firstIdx; time++)
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
				gradient = gradient / 
					make_float2
				(
					 2.0f* solverOptions.gridDiameter[0] / (dispersionOptions.gridSize_2D[0]-1),
					 2.0f * solverOptions.gridDiameter[2] /( dispersionOptions.gridSize_2D[1]-1)
				);


				gradient = { normalize(make_float3(1.0,gradient.x, gradient.y)).y, normalize(make_float3(1.0,gradient.x, gradient.y)).z };
			}




			float4 texel = { 0,0,0,0 };
			texel.x = observable.ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y,time)).x;
			texel.y = gradient.x;
			texel.z = gradient.y;
			texel.w = observable.ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(index_x, index_y, time)).w;

			surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * index_x, index_y,time);

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
				float2 gradient = { 0.0f,0.0f };


				gradient = observable.GradientFluctuatuionAtXZ(heightFieldSurface3D, make_int3(z,index, t), gridSize);
				gradient = gradient /
					make_float2
					(
						2.0f * solverOptions.gridDiameter[0] / (gridSize.x - 1),
						2.0f * solverOptions.gridDiameter[2] / (gridSize.z - 1)
					);

				float3 gradient3D = normalize(make_float3(1.0, gradient.x, gradient.y));
				gradient = { gradient3D.y, gradient3D.z };
			

				float4 texel = observable.ValueAtXYZ_Surface_float4(heightFieldSurface3D, make_int3(z, index, t));

				texel.z = gradient.x;
				texel.w = gradient.y;

				surf3Dwrite(texel, heightFieldSurface3D, sizeof(float4) * z, index, t);


			}

		}


	}
}
