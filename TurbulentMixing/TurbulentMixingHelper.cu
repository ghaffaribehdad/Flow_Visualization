#include "TurbulentMixingHelper.h"
#include "..//Cuda/helper_math.h"

__global__ void createTKE(cudaSurfaceObject_t s_mixing, cudaTextureObject_t v_field, SolverOptions solverOptions, TurbulentMixingOptions turbulentMixingOptions)
{
	// Convert block and thread ID to linear index
	int index = CUDA_INDEX;

	//if (index < solverOptions.gridSize[1] * solverOptions.gridSize[2])
	if (index <1024 * 764)
	{
		// Extract pixel position
		int2 pixel = { 0,0 };
		//pixel.y = index / solverOptions.gridSize[2]; // Y -> 503
		pixel.y = index / 1024; // Y -> 503
		//pixel.x = index - pixel.y * solverOptions.gridSize[2]; // Z -> 2048
		pixel.x = index - pixel.y * 1024; // Z -> 2048

		// Store the value of the streamwise velocity at that pixel

		float streamwiseVel = tex2D<float4>(v_field, pixel.y, pixel.x).x;

		float4 value = { 1,1,1,0 };

		//float ratio = linearCreation(503, pixel.y, turbulentMixingOptions.linearCreationRatio);
		float ratio = 1.0f;

		surf2Dwrite(value, s_mixing, sizeof(float4) * pixel.x, pixel.y);

	}
}