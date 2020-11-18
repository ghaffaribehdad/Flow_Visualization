#pragma once
#include "cuda_runtime.h"
#include <vector>
#include "../cudaCompress/src/examples/CompressVolume.h"

typedef unsigned int uint;

float * decompress(int3 size, uint * h_data, const float & Quant_step, GPUResources::Config & config, GPUResources & shared, CompressVolumeResources & res, size_t & bufferSize);
void releaseGPUResources(float * dp_field);


struct DecompressResources
{

	GPUResources::Config config;
	GPUResources shared;
	CompressVolumeResources res;
	uint huffmanBits = 0;
	void initializeDecompressionResources(int3 size);
	void releaseDecompressionResources();


};