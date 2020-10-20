
#include "VolumeTexture.h"
#include <string>







bool VolumeTexture3D::initialize
(
	const int3 & dimension,
	bool normalizedCoords,
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
	texDesc.normalizedCoords = normalizedCoords;

	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}



bool VolumeTexture3D::initialize_devicePointer
(
	const int3 & dimension,
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureAddressMode addressMode_z,
	cudaTextureFilterMode _cudaTextureFilterMode
)
{

	cudaExtent extent = make_cudaExtent(dimension.x, dimension.y, dimension.z);

	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMalloc3DArray(&this->cuArray_velocity, &channelFormatDesc, extent));



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_field, extent.width * sizeof(float4), extent.width, extent.height);
	cpyParams.dstArray = this->cuArray_velocity;
	cpyParams.kind = cudaMemcpyDeviceToDevice;
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
	texDesc.normalizedCoords = normalizedCoords;

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
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureAddressMode addressMode_z,
	cudaTextureFilterMode _cudaTextureFilterMode
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
	texDesc.normalizedCoords = normalizedCoords;
	texDesc.filterMode = _cudaTextureFilterMode;
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
	gpuErrchk(cudaDestroyTextureObject(this->t_field));
	gpuErrchk(cudaFreeArray(this->cuArray_velocity));

}





bool VolumeTexture2D::initialize
(
	const int2 & gridSize,
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureFilterMode _cudaTextureFilterMode
)
{


	// Allocate 2D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMallocArray(&cuArray_velocity, &channelFormatDesc, gridSize.x, gridSize.y));


	gpuErrchk(cudaMemcpy2DToArray(this->cuArray_velocity,0,0,h_field, gridSize.x * sizeof(float4), gridSize.x *sizeof(float4), gridSize.y,cudaMemcpyHostToDevice));



	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = normalizedCoords;
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
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureFilterMode _cudaTextureFilterMode

)
{

	
	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = normalizedCoords;
	texDesc.filterMode = _cudaTextureFilterMode;
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






bool VolumeTexture1D::initialize
(
	size_t width,
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureFilterMode _cudaTextureFilterMode

)
{

	// Allocate 2D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float>();
	gpuErrchk(cudaMallocArray(&cuArray_velocity, &channelFormatDesc, width));


	gpuErrchk(cudaMemcpy2DToArray(this->cuArray_velocity, 0, 0, h_field, width * sizeof(float), width * sizeof(float), 0, cudaMemcpyHostToDevice));

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
	texDesc.normalizedCoords = normalizedCoords;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


bool VolumeTexture1D::initialize_array
(

	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureFilterMode _cudaTextureFilterMode

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

	texDesc.filterMode = _cudaTextureFilterMode;
	texDesc.normalizedCoords = normalizedCoords;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}



void VolumeTexture1D::release()
{

	cudaFreeArray(this->cuArray_velocity);
	cudaDestroyTextureObject(this->t_field);

}


// Explicit declaration
template VolumeTexture3D_T<float4>;
template VolumeTexture3D_T<float3>;
template VolumeTexture3D_T<float>;

template<typename T>
bool VolumeTexture3D_T<T>::initialize
(
	const int3 & dimension,
	bool normalizedCoords,
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

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_field, extent.width * sizeof(T), extent.width, extent.height);
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
	texDesc.normalizedCoords = normalizedCoords;

	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}


template<typename T>
bool VolumeTexture3D_T<T>::initialize_array
(
	bool normalizedCoords,
	cudaTextureAddressMode addressMode_x,
	cudaTextureAddressMode addressMode_y,
	cudaTextureAddressMode addressMode_z,
	cudaTextureFilterMode _cudaTextureFilterMode
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
	texDesc.normalizedCoords = normalizedCoords;
	texDesc.filterMode = _cudaTextureFilterMode;
	texDesc.addressMode[0] = addressMode_x;
	texDesc.addressMode[1] = addressMode_y;
	texDesc.addressMode[2] = addressMode_z;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_field, &resDesc, &texDesc, NULL));

	return true;

}

template<typename T>
void VolumeTexture3D_T<T>::release()
{
	cudaFreeArray(this->cuArray_velocity);
	cudaDestroyTextureObject(this->t_field);

}
