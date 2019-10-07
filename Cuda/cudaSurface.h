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