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

	cudaArray_t& getArrayRef()
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
	cudaArray_t cuArray;
	cudaExtent extent = { 0,0,0 };
	bool initialized = false;
	cudaChannelFormatDesc channelFormatDesc;



public:


	bool initialize(const int & x, const int & y, const int & z)
	{

		this->extent = make_cudaExtent(x, y, z);

		// Create Format description based on the template typename
		this->channelFormatDesc = cudaCreateChannelDesc<T>();

		// Allocate 3D Array
		gpuErrchk(cudaMalloc3DArray(&this->cuArray, &channelFormatDesc, extent))

			// set initialization status to true
			this->initialized = true;

		return true;
	}

	bool memoryCopy(float * h_field)
	{
		if (this->initialized == true)
		{
			// set copy parameters to copy from velocity field to array
			cudaMemcpy3DParms cpyParams = { 0 };

			cpyParams.srcPtr = make_cudaPitchedPtr((void*)h_field,  sizeof(T) * extent.width, extent.width, extent.height);
			cpyParams.dstArray = this->cuArray;
			cpyParams.kind = cudaMemcpyHostToDevice;
			cpyParams.extent = extent;

			// Copy velocities to 3D Array
			gpuErrchk(cudaMemcpy3D(&cpyParams))
			{
				return false;
			}
			
			return true;
		}
		else
		{
			return false;
		}

	}


	cudaArray_t getArray()
	{
		return this->cuArray;
	}

	cudaArray_t & getArrayRef()
	{
		return cuArray;
	}

	void release()
	{
		gpuErrchk(cudaFreeArray(this->cuArray));
	}

};