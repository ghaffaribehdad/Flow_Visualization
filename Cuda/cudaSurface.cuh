#pragma once
#include "cuda_runtime.h"

class CudaSurface
{
private:

	cudaSurfaceObject_t surfaceObject = 0;

	cudaArray* cuInputArray;

	int width = 0;
	int height = 0;

public:

	bool initializeSurface();
	bool destroySurface();

	cudaSurfaceObject_t getSurfaceObject();
};