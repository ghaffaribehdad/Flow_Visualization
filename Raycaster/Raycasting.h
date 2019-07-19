#pragma once
#include "Ray.cuh"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/CudaSurface.cuh"
#include "cuda_runtime.h"
#include "BoundingBox.cuh"

__global__ void boundingBoxRendering(Ray* d_Ray, BoundingBox* d_boundingBox,cudaSurfaceObject_t raycastingSurface, int rays);

struct Raycasting_desc
{
	float3 eyePos;
	float3 viewDir;
	float3 upDir;
	float FOV_deg;

	int width;
	int height;

	float3 gridDiameter;
	
	// for now this is fixed 
	float3 gridCenter = { 0,0,0 };
};


class Raycasting
{

public:


	__host__ bool initialize();
	__host__ bool release();
	__host__ void Rendering();


	__host__ void setRaycastingDec(Raycasting_desc & _raycasting_desc)
	{
		this->raycasting_desc = _raycasting_desc;
	}
	__host__ void setRaycastingSurface(CudaSurface * _raycastingSurface)
	{
		this->raycastingSurface = _raycastingSurface;
	}


private:

	int maxThreadBlock = 256;
	size_t rays = 0;
	
	__host__ bool initilizeBoundingBox();
	__host__ bool initializeVolumeTexuture();

	Raycasting_desc raycasting_desc;

	VolumeTexture* volumeTexture;
	CudaSurface * raycastingSurface;

	BoundingBox * h_boundingBox;
	BoundingBox * d_boundingBox;

	Ray* h_Ray;
	Ray* d_Ray;


};