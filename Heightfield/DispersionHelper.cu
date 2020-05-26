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
	int index = CUDA_INDEX;
	
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




__global__ void fetch_ftle_height
(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_height,
	float * d_ftle,
	SolverOptions solverOptions,
	DispersionOptions dispersionOptions,
	int timestep
)
{
	int index = CUDA_INDEX;


	if (index < dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1])
	{	
		int2 dim = make_int2(dispersionOptions.gridSize_2D[0], dispersionOptions.gridSize_2D[1]);
		int2 pixel = { 0,0 };
		pixel = IndexToPixel(index,dim);


		d_height[index] = ValueAtXYZ_float4(t_height, make_float3(pixel.x, pixel.y, timestep)).x;
		d_ftle[index] = ValueAtXYZ_float4(t_ftle, make_float3(pixel.x, pixel.y, timestep)).x;

	}
}



__global__ void textureMean(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_mean_height,
	float * d_mean_ftle,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{
	int index = CUDA_INDEX;
	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	if (index < gridPoints)
	{
		int index_y = index / dispersionOptions.gridSize_2D[1]; // 
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);

		float height = 0;
		float ftle = 0;

		for(int time = 0; time < solverOptions.timeSteps; time++)
		{
			// extract the values
			height = ValueAtXYZ_float4(t_height, make_float3(index_x, index_y, time)).x;
			ftle = ValueAtXYZ_float4(t_ftle, make_float3(index_x, index_y, solverOptions.timeSteps - 1)).x;

			// add them to the mean values
			atomicAdd_system(&d_mean_height[time], height / (float)gridPoints);
			atomicAdd_system(&d_mean_ftle[time], ftle / (float)gridPoints );

		}

	}
}


__global__ void pearson_terms(
	cudaTextureObject_t t_height,
	cudaTextureObject_t t_ftle,
	float * d_mean_height,
	float * d_mean_ftle,
	float * d_pearson_cov,
	float * d_pearson_var_ftle,
	float * d_pearson_var_height,
	DispersionOptions dispersionOptions,
	SolverOptions solverOptions
)
{
	int index = CUDA_INDEX;
	int gridPoints = dispersionOptions.gridSize_2D[0] * dispersionOptions.gridSize_2D[1];

	if (index < gridPoints)
	{
		int index_y = index / dispersionOptions.gridSize_2D[1]; // 
		int index_x = index - (index_y * dispersionOptions.gridSize_2D[1]);

		float covariance = 0;
		float variance_ftle = 0;
		float variance_height = 0;

		for (int time = 0; time < solverOptions.timeSteps; time++)
		{
			// calculate terms of pearson
			covariance = (ValueAtXYZ_float4(t_height, make_float3(index_x, index_y, time)).x - d_mean_height[time]) * (ValueAtXYZ_float4(t_ftle, make_float3(index_x, index_y, solverOptions.timeSteps - 1)).x - d_mean_ftle[time]);
			variance_height =powf( ValueAtXYZ_float4(t_height, make_float3(index_x, index_y, time)).x - d_mean_height[time], 2.0f);
			variance_ftle =powf( ValueAtXYZ_float4(t_ftle, make_float3(index_x, index_y, solverOptions.timeSteps - 1)).x - d_mean_ftle[time],2.0f);
			

			atomicAdd_system(&d_pearson_cov[time], covariance);
			atomicAdd_system(&d_pearson_var_ftle[time], variance_ftle);
			atomicAdd_system(&d_pearson_var_height[time], variance_height);
			

		}

	}
}



__global__ void pearson(
	float * d_pearson_cov,
	float * d_pearson_var_ftle,
	float * d_pearson_var_height,
	SolverOptions solverOptions
)
{
	int index = threadIdx.x;

	d_pearson_cov[index] = d_pearson_cov[index] / sqrtf(d_pearson_var_ftle[index] * d_pearson_var_height[index]);
	
}