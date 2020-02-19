#include "TurbulentMixingHelper.h"
#include "..//Cuda/helper_math.h"

__global__ void createTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions)
{
	// Convert block and thread ID to linear index
	int index = CUDA_INDEX;

	if (index < solverOptions.gridSize[1] * solverOptions.gridSize[2])
	{
		// Extract pixel position
		int2 pixel = { 0,0 };
		pixel.x = index / solverOptions.gridSize[2]; // Y -> 503
		pixel.y = index - pixel.x * solverOptions.gridSize[2]; // Z -> 2048

		// calculates the creation ratio
		float ratio = linearCreation(503, pixel.x, turbulentMixingOptions.linearCreationRatio);

		//Retrieve the old value from the surface
		float4 texel;
		surf2Dread(&texel, s_mixing, pixel.x * sizeof(float4), pixel.y);

		// Read the value of the streamwise velocity at that pixel
		float streamwiseVel = tex2D<float4>(v_field, pixel.x, pixel.y).x;

		// texel.x stores the positive momentum and texel.y the negative ones
		if (streamwiseVel > 0)
		{
			texel.x += ratio * streamwiseVel;
		}
		else
		{
			texel.y +=  ratio * fabsf(streamwiseVel);
		}


		surf2Dwrite(texel, s_mixing, sizeof(float4) * pixel.x, pixel.y);

	}
}



__global__ void dissipateTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions)
{
	// Convert block and thread ID to linear index
	int index = CUDA_INDEX;

	if (index < solverOptions.gridSize[1] * solverOptions.gridSize[2])
	{
		// Extract pixel position
		int2 pixel = { 0,0 };
		pixel.x = index / solverOptions.gridSize[2]; // Y -> 503
		pixel.y = index - pixel.x * solverOptions.gridSize[2]; // Z -> 2048

		// calculates the creation ratio
		float ratio = linearDissipation(503, pixel.x, turbulentMixingOptions.linearCreationRatio);

		//Retrieve the old value from the surface
		float4 texel;
		surf2Dread(&texel, s_mixing, pixel.x * sizeof(float4), pixel.y);
	
		// Reduce the momentum base on the ratio
		texel = texel * ratio;

		// Write back the reduced momentum
		surf2Dwrite(texel, s_mixing, sizeof(float4) * pixel.x, pixel.y);

	}
}


__global__ void advectTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field_0, cudaTextureObject_t v_field_1, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions)
{
	// Convert block and thread ID to linear index
	int index = CUDA_INDEX;

	if (index < solverOptions.gridSize[1] * solverOptions.gridSize[2])
	{
		// Extract pixel position
		int2 pixel = { 0,0 };
		pixel.x = index / solverOptions.gridSize[2]; // Y -> 503
		pixel.y = index - pixel.x * solverOptions.gridSize[2]; // Z -> 2048

		// calculates the creation ratio
		float ratio = linearDissipation(503, pixel.x, turbulentMixingOptions.linearCreationRatio);

		//Retrieve the old value from the surface
		float4 texel;
		surf2Dread(&texel, s_mixing, pixel.x * sizeof(float4), pixel.y);

		// Reduce the momentum base on the ratio
		texel = texel * ratio;

		// Write back the reduced momentum
		surf2Dwrite(texel, s_mixing, sizeof(float4) * pixel.x, pixel.y);

	}
}