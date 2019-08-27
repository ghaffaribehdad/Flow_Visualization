
#pragma once

#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/CudaSurface.cuh"
#include "cuda_runtime.h"
#include "BoundingBox.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Volume/Volume_IO.h"
#include "..//SolverOptions.h"
#include <vector>
#include "texture_fetch_functions.h"



template <typename Observable>
__global__ void CudaIsoSurfacRenderer(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);

struct Raycasting_desc
{
	float3 eyePos = { 0,0,0 };
	float3 viewDir = { 0,0,0 };
	float3 upDir = { 0,0,0 };
	float FOV_deg = 0.0f;

	int width = 0;
	int height = 0;

	float3 gridDiameter = { 0,0,0 };
	int3 gridSize = { 0,0,0 };

	float* field = nullptr;

	// for now this is fixed 
	float3 gridCenter = { 0,0,0 };

	SolverOptions * solverOption;
};


class Raycasting
{

public:


	__host__ bool initialize();
	__host__ bool release();
	__host__ void Rendering();


	__host__ void setRaycastingDec(Raycasting_desc& _raycasting_desc)
	{
		this->raycasting_desc = _raycasting_desc;
	}
	__host__ void setRaycastingSurface(CudaSurface* _raycastingSurface)
	{
		this->raycastingSurface = _raycastingSurface;
	}


private:

	bool fileLoaded = false;
	bool fileChanged = false;

	unsigned int maxBlockDim = 32;

	size_t rays = 0;

	__host__ bool initilizeBoundingBox();
	__host__ bool initializeIO();
	__host__ bool initializeVolumeTexuture();


	Raycasting_desc raycasting_desc;

	VolumeTexture volumeTexture;
	CudaSurface* raycastingSurface;

	Volume_IO volume_IO;
	BoundingBox * d_BoundingBox;

};

