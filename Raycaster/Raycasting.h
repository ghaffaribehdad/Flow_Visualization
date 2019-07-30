

#ifndef RAYCASTING_H
#define RAYCASINTG_H



#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/CudaSurface.cuh"
#include "cuda_runtime.h"
#include "BoundingBox.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Volume/Volume_IO.h"
#include "..//SolverOptions.h"
#include <vector>
#include "texture_fetch_functions.h"


//__global__ void boundingBoxRendering(BoundingBox* d_boundingBox,cudaSurfaceObject_t raycastingSurface,cudaTextureObject_t field1, int rays);
__global__ void isoSurfaceVelocityMagnitude(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);


struct Raycasting_desc
{
	float3 eyePos;
	float3 viewDir;
	float3 upDir;
	float FOV_deg;

	int width;
	int height;

	float3 gridDiameter;
	int3 gridSize;

	float* field;

	// for now this is fixed 
	float3 gridCenter = { 0,0,0 };

	SolverOptions solverOption;
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



#endif // !RAYCASTING_H
