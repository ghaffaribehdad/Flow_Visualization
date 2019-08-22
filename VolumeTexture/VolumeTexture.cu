
#include "VolumeTexture.h"
#include <string>






cudaTextureObject_t VolumeTexture::initialize()
{
	// Cuda 3D array of velocities
	cudaArray_t cuArray_velocity;


	// define the size of the velocity field
	cudaExtent extent =
	{
		static_cast<size_t>(this->solverOptions->gridSize[0]),
		static_cast<size_t>(this->solverOptions->gridSize[1]),
		static_cast<size_t>(this->solverOptions->gridSize[2])
	};


	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(&cuArray_velocity, &channelFormatDesc, extent, 0);



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_field, extent.width * sizeof(float4), extent.height, extent.depth);
	cpyParams.dstArray = cuArray_velocity;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cpyParams.extent = extent;


	// Copy velocities to 3D Array
	gpuErrchk(cudaMemcpy3D(&cpyParams));
	// might need sync before release the host memory

	// Release the Volume while it is copied on GPU
	//this->volume_IO.release();


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc resViewDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&resViewDesc, 0, sizeof(resViewDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return t_field;

}
void VolumeTexture::release()
{
	cudaDestroyTextureObject(this->t_field);
}



