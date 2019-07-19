#include "cudaSurface.cuh"

bool CudaSurface::initializeSurface()
{


	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(16, 16, 16, 16,
			cudaChannelFormatKindUnsigned);


	//cudaMallocArray(&this->cuInputArray, &channelDesc, this->width, this->height,\
		//cudaArraySurfaceLoadStore);



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

	// Free device memory
	//cudaFreeArray(cuInputArray);

	return true;
}