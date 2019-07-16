#pragma once
#include "cuda_runtime.h"
#include <string.h>

class CudaSurface
{
private:

	cudaSurfaceObject_t surfaceObject = 0;

	cudaArray* cuInputArray;

	int width = 0;
	int height = 0;
	CudaSurface();

public:

	CudaSurface(const int& _width, const int& _height)
	{
		this->width = _width;
		this->height = _height;
	}

	cudaSurfaceObject_t getSurfaceObject()
	{
		return this->surfaceObject;
	}
	bool initializeSurface();
	bool destroySurface();

};