
#include "VolumeTexture.h"
#include <string>






bool VolumeTexture3D::initialize
(
	cudaTextureAddressMode addressMode_x ,
	cudaTextureAddressMode addressMode_y ,
	cudaTextureAddressMode addressMode_z,
	cudaTextureFilterMode _cudaTextureFilterMode
)
{
	if (this->solverOptions == nullptr)
	{
		return false;
	}

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
	
	texDesc.filterMode = _cudaTextureFilterMode;
	if (_cudaTextureFilterMode == cudaFilterModeLinear)
	{
		texDesc.normalizedCoords = true;
	}
	else
	{
		texDesc.normalizedCoords = false;
	}

	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


bool VolumeTexture3D::initialize
(
	int3 dimension,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureAddressMode addressMode_z,
	cudaTextureFilterMode _cudaTextureFilterMode
)
{

	cudaExtent extent = make_cudaExtent(dimension.x, dimension.y, dimension.z);

	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(&this->cuArray_velocity, &channelFormatDesc, extent);



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_field, extent.width * sizeof(float4), extent.width, extent.height);
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
	texDesc.filterMode = _cudaTextureFilterMode;
	if (_cudaTextureFilterMode == cudaFilterModeLinear)
	{
		texDesc.normalizedCoords = true;
	}
	else
	{
		texDesc.normalizedCoords = false;
	}
	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


bool VolumeTexture3D::initialize_array
(
	int3 dimension,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureAddressMode addressMode_z
)
{

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

	return true;

}


void VolumeTexture3D::release()
{
	cudaFreeArray(this->cuArray_velocity);
	cudaDestroyTextureObject(this->t_field);

}





bool VolumeTexture2D::initialize
(
	size_t width,
	size_t height,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y
)
{


	// Allocate 2D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMallocArray(&cuArray_velocity, &channelFormatDesc, width, height));


	gpuErrchk(cudaMemcpy2DToArray(this->cuArray_velocity,0,0,h_field, width * sizeof(float4),width*sizeof(float4),height,cudaMemcpyHostToDevice));



	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


bool VolumeTexture2D::initialize_array
(
	size_t width,
	size_t height,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y
)
{


	// Allocate 2D Array




	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


void VolumeTexture2D::release()
{

	cudaFreeArray(this->cuArray_velocity);
	cudaDestroyTextureObject(this->t_field);

}