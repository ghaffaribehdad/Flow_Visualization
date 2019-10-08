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