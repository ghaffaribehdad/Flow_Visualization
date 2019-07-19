#pragma once
#include "Ray.cuh"
#include "..//VolumeTexture/VolumeTexture.h"


struct Raycasting_desc
{
	float3 eyePos;
	float3 viewDir;
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


	__host__ void setRaycastingDec(Raycasting_desc & _raycasting_desc)
	{
		this->raycasting_desc = _raycasting_desc;
	}
	__host__ void setRaycastingSurface(cudaSurfaceObject_t _raycastingSurface)
	{
		this->raycastingSurface = _raycastingSurface;
	}


private:

	Raycasting_desc raycasting_desc;

	VolumeTexture* volumeTexture;
	cudaSurfaceObject_t raycastingSurface;

	BoundingBox h_boundingBox;
	BoundingBox d_boundingBox;

	Ray* h_Ray;
	Ray* d_Ray;


};