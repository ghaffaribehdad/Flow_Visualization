#pragma once
#include "cuda_runtime.h"
#include "..//ErrorLogger/ErrorLogger.h"

template <typename T>
class CudaArray_2D
{
public:

	bool initialize()
	{
		// Create Format description based on the template typename
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<T>();


		// Allocate CUDA array in device memory
		gpuErrchk(cudaMallocArray(&this->cuArray, &channelFormatDesc,width,height));

		return true;
	}

	void setDimension(const size_t& _width, const size_t& _height)
	{
		this->width		= _width;
		this->height	= _height;
	}

	cudaArray_t getArray()
	{
		return cuArray;
	}

	void release()
	{
		gpuErrchk(cudaFreeArray(this->cuArray));
	}

private:
	cudaArray_t cuArray = nullptr;
	size_t width	= 0;
	size_t height	= 0;
};


template <typename T>
class CudaArray_3D
{



private:
	cudaArray_t cuArray = nullptr;
	size_t width = 0;
	size_t height = 0;
	size_t depth = 0;
	cudaExtent extent = { 0,0,0 };
	bool initialized = false;
	cudaChannelFormatDesc channelFormatDesc;



public:

	bool initialize()
	{

		this->extent = make_cudaExtent(this->width, this->height, this->depth);

		// Allocate 3D Array
		cudaMalloc3DArray(&this->cuArray, &channelFormatDesc, extent);


		// Create Format description based on the template typename
		this->channelFormatDesc = cudaCreateChannelDesc<T>();


		// Allocate CUDA array in device memory
		gpuErrchk(cudaMallocArray(&this->cuArray, &channelFormatDesc, width, height));

		// set initialization status to true
		this->initialized = true;

		return true;
	}

	bool memoeryCopy(void * h_field)
	{
		if (this->initialized == true)
		{
			// set copy parameters to copy from velocity field to array
			cudaMemcpy3DParms cpyParams = { 0 };

			cpyParams.srcPtr = make_cudaPitchedPtr((void*)h_field, extent.width * sizeof(float4), extent.width, extent.height);
			cpyParams.dstArray = this->cuArray;
			cpyParams.kind = cudaMemcpyHostToDevice;
			cpyParams.extent = extent;

			// Copy velocities to 3D Array
			gpuErrchk(cudaMemcpy3D(&cpyParams));

			return true;
		}
		else
		{
			return false;
		}

	}

	void setDimension(const int& _width, const int& _height, int& _depth)
	{
		this->width = _width;
		this->height = _height;
		this->depth = _depth;
	}

	cudaArray_t getArray()
	{
		return cuArray;
	}

	void release()
	{
		gpuErrchk(cudaFreeArray(this->cuArray));
	}

};