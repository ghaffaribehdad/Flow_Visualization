#pragma once
#include "cuda_runtime.h"
#include "..//ErrorLogger/ErrorLogger.h"

template <typename T>
class CudaArray_3D
{
public:

	bool initialize()
	{
		// Create Format description based on the template typename
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<T>();

		// Create Extent according to the dimensions of the array
		cudaExtent extent = make_cudaExtent(this->width, this->height, this->depth);

		// Allocate CUDA array in device memory
		gpuErrchk(cudaMalloc3DArray(&this->cuArray, &channelFormatDesc, extent));

		return true;
	}

	void setDimension(const size_t& _width, const size_t& _height, const size_t& _depth)
	{
		this->width		= _width;
		this->height	= _height;
		this->depth		= _depth;
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
	cudaArray_t cuArray;
	size_t width;
	size_t height;
	size_t depth;



};