#pragma once

#include "cuda_runtime.h"

typedef unsigned int uint;

//return the index of an element in the sequential form based on its index in tensor form (4D)
__host__ __device__ inline uint linearIndex(uint x, uint y, uint z, uint w, int4 dim)
{
	uint index = 0;
	index += x * dim.y *dim.z * dim.w;
	index += y * dim.z * dim.w;
	index += z * dim.w;
	index += w;

	return index;
}

//return the index of an element in the sequential form based on its index in tensor form (3D)
__host__ __device__ inline uint linearIndex(uint x, uint y, uint z, int3 dim)
{
	uint index = 0;
	index += x * dim.y *dim.z;
	index += y * dim.z;
	index += z;


	return index;
}