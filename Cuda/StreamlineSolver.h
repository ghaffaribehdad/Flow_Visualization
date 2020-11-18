#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"
#include <vector>
#include <stdio.h>
#include "../Particle/Particle.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"


class StreamlineSolver : public CUDASolver
{

public:

	__host__ bool solve() override;
	__host__ bool loadVolumeTexture();
	__host__ bool releaseVolumeIO();
	__host__ bool releaseVolumeTexture();
	__host__ bool release();

private:

	Particle* d_particles;
	VolumeTexture3D volumeTexture;

	__host__ bool InitializeVorticityTexture();


	CudaArray_3D<float4>	a_Measure;
	CudaSurface				s_Measure;
	cudaTextureObject_t		t_measure;
	
	// https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	// Reference to Velocity Field as a Texture 


	float3* result;

};







