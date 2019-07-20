#include "Raycasting.h"

__host__ bool Raycasting::initialize()
{
	this->rays = this->raycasting_desc.width * this->raycasting_desc.height;
	h_boundingBox = new BoundingBox();
	this->h_boundingBox->setEyePos(this->raycasting_desc.eyePos);
	this->h_boundingBox->setViewDir(this->raycasting_desc.viewDir);
	this->h_boundingBox->setUpVect(this->raycasting_desc.upDir);

	this->h_boundingBox->setResolution(this->raycasting_desc.width, this->raycasting_desc.height);
	this->h_boundingBox->setGridDiametr(this->raycasting_desc.gridDiameter);

	this->h_boundingBox->initialize();

	cudaMalloc((void**)&d_boundingBox, sizeof(BoundingBox));
	cudaMemcpy(d_boundingBox, h_boundingBox, sizeof(BoundingBox), cudaMemcpyHostToDevice);

	delete h_boundingBox;

	
	return true;

}

__host__ bool Raycasting::release()
{
	cudaFree(d_boundingBox);

	return true;
}

__host__ void Raycasting::Rendering()
{
	int blocks = (rays % (maxThreadBlock )) == 0 ? rays / maxThreadBlock : (rays / maxThreadBlock ) + 1;

	//boundingBoxRendering <<< blocks, maxThreadBlock >>> (this->d_boundingBox, this->raycastingSurface->getSurfaceObject(), this->rays);
	boundingBoxRendering << < blocks, maxThreadBlock >> > (this->d_boundingBox, this->raycastingSurface->getSurfaceObject(), this->rays);

	cudaDeviceSynchronize();
}


__global__ void boundingBoxRendering(BoundingBox* d_boundingBox, cudaSurfaceObject_t raycastingSurface, int rays)
{
	// Calculate surface coordinates
	int thread = threadIdx.x;
	
	int index = blockIdx.x * blockDim.x + thread;

	if (index < rays)
	{


		int2 pixel = { 0,0 };
		pixel.x = index / d_boundingBox->getResolution().y;
		pixel.y = index - (pixel.x * d_boundingBox->getResolution().y);


		float3 pixelPos = d_boundingBox->pixelPosition(pixel.x, pixel.y);
		float2 NearFar = d_boundingBox->findIntersections(pixelPos);

		if (NearFar.x != -1)
		{
			float4 color = { 1,1,0,0 };
			float rgba = DecodeFloatRGBA(color);
			surf2Dwrite(rgba, raycastingSurface, sizeof(float) * pixel.x, pixel.y);
		}

	}
}