
#include "VolumeTexture.h"
#include "..//ErrorLogger.h"

void VolumeTexture::setGridDiameter(const float3& _gridDiamter)
{
	this->gridDiameter = _gridDiamter;
}

void VolumeTexture::setGridSize(const int3& _gridSize)
{
	this->gridSize = _gridSize;
}

void VolumeTexture::setField(float* _h_field)
{
	this->h_field = _h_field;
}

void VolumeTexture::initialize()
{
	// Cuda 3D array of velocities
	cudaArray_t cuArray_velocity;


	// define the size of the velocity field
	cudaExtent extent =
	{
		static_cast<size_t>(this->gridSize.x),
		static_cast<size_t>(this->gridSize.y),
		static_cast<size_t>(this->gridSize.z)
	};


	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMalloc3DArray(&cuArray_velocity, &channelFormatDesc, extent, 0));



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
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

}
void VolumeTexture::release()
{
	gpuErrchk(cudaDestroyTextureObject(this->t_field));
}


const int3 & VolumeTexture::getGridSize() const
{
	return this->gridSize;
}
const float3& VolumeTexture::getGridDiameter() const
{
	return this->gridDiameter;
}


__device__ float4 VolumeTexture::fetch(float3 index)
{
	return tex3D<float4>(this->t_field, index.x, index.y, index.z);
}