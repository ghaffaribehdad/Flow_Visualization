#pragma once
#include "cuda_runtime.h"
#include <string.h>

class CudaSurface
{
private:

	cudaSurfaceObject_t surfaceObject;

	cudaArray_t cuInputArray = NULL;

public:


	cudaSurfaceObject_t getSurfaceObject()
	{
		return this->surfaceObject;
	}

	cudaSurfaceObject_t& getSurfaceObjectRef()
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