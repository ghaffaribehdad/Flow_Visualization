
#include "VolumeTexture.h"
#include <string>






cudaTextureObject_t VolumeTexture::initialize
(
	cudaTextureAddressMode addressMode_x ,
	cudaTextureAddressMode addressMode_y ,
	cudaTextureAddressMode addressMode_z
)
{

	cudaExtent extent = make_cudaExtent(this->solverOptions->gridSize[0], this->solverOptions->gridSize[1], this->solverOptions->gridSize[2]);

	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(&this->cuArray_velocity , &channelFormatDesc, extent);



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_field,extent.width * sizeof(float4),extent.width, extent.height);
	cpyParams.dstArray = this->cuArray_velocity;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cpyParams.extent = extent;
	
	// Copy velocities to 3D Array
	gpuErrchk(cudaMemcpy3D(&cpyParams));
	// might need sync before release the host memory


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc resViewDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&resViewDesc, 0, sizeof(resViewDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return t_field;

}
void VolumeTexture::release()
{
	cudaFreeArray(this->cuArray_velocity);
	cudaDestroyTextureObject(this->t_field);
}



