#include "Raycasting.h"

__host__ bool Raycasting::initialize()
{
	h_boundingBox = new BoundingBox();
	this->h_boundingBox->setEyePos(this->raycasting_desc.eyePos);
	this->h_boundingBox->setViewDir(this->raycasting_desc.viewDir);
	this->h_boundingBox->setResolution(this->raycasting_desc.width, this->raycasting_desc.height);
	this->h_boundingBox->setUpVect(this->raycasting_desc.upDir);
	this->h_boundingBox->setGridDiametr(this->raycasting_desc.gridDiameter);

	this->h_boundingBox->initialize();

	cudaMalloc((void**)&d_boundingBox, sizeof(BoundingBox));
	cudaMemcpy(d_boundingBox, h_boundingBox, sizeof(BoundingBox), cudaMemcpyHostToDevice);


	this->rays = static_cast<size_t>(raycasting_desc.width) * static_cast<size_t>(raycasting_desc.width);
	this->h_Ray = new Ray[rays];
	
	cudaMalloc((void**)& d_Ray, sizeof(Ray) * rays);
	cudaMemcpy(this->d_Ray, h_Ray, sizeof(Ray) * rays, cudaMemcpyHostToDevice);
	

	return true;

}

__host__ bool Raycasting::release()
{
	cudaFree(d_boundingBox);
	cudaFree(d_Ray);

	delete[] h_Ray;
	delete h_boundingBox;

	return true;
}

__host__ void Raycasting::Rendering()
{
	int blocks = (rays % maxThreadBlock) == 0 ? rays / maxThreadBlock : (rays / maxThreadBlock) + 1;

	boundingBoxRendering <<< blocks, maxThreadBlock >>> (this->d_Ray, this->d_boundingBox, this->raycastingSurface->getSurfaceObject(), this->rays);
	
}


__global__ void boundingBoxRendering(Ray* d_Ray, BoundingBox* d_boundingBox, cudaSurfaceObject_t raycastingSurface, int rays)
{
	// Calculate surface coordinates
	unsigned int x = threadIdx.x;

	unsigned int index = x * blockDim.x + x;

	if (index < rays)
	{

		d_Ray[index].setPixel(index / d_boundingBox->getResolution().x, index - (index / d_boundingBox->getResolution().x) * d_boundingBox->getResolution().x);
		//d_Ray[index].boundingBoxIntersection(d_boundingBox);

	/*	float2 intersection = d_Ray[index].getIntersection();

		if (intersection.x != -1)
		{
			float4 texel = { 1, 1,1,0 };
			printf("hit!\t");
			surf2Dwrite(texel, raycastingSurface, d_Ray[index].pixel.x, d_Ray[index].pixel.y);
		}*/

	}
}