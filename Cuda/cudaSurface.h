#pragma once
#include "cuda_runtime.h"
#include <string.h>

class CudaSurface
{
private:

	cudaSurfaceObject_t surfaceObject = 0;

	cudaArray_t cuInputArray = NULL;

	int width = 0;
	int height = 0;

public:

	void setDimensions(const int& _width, const int& _height)
	{
		this->width		= _width;
		this->height	= _height;
	}
	cudaSurfaceObject_t getSurfaceObject()
	{
		return this->surfaceObject;
	}

	void setInputArray(cudaArray_t & _cuInputArray)
	{
		this->cuInputArray = _cuInputArray;
	}

	bool initializeSurface();
	bool destroySurface();

};