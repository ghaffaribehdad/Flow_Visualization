#include "cudaSurface.h"

bool CudaSurface::initializeSurface()
{
	// Specify surface
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	// Create the surface objects
	resDesc.res.array.array = this->cuInputArray;

	cudaCreateSurfaceObject(&surfaceObject, &resDesc);

	return true;
}


bool CudaSurface::destroySurface()
{

	// Destroy surface objects
	cudaDestroySurfaceObject(surfaceObject);

	

	return true;
}
