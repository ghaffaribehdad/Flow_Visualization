#include "Compression.h"
#include "../ErrorLogger/ErrorLogger.h"


#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cudaCompress/Instance.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/util/Bits.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
#include <cudaCompress/util/YCoCg.h>
#include <cudaCompress/Timing.h>
using namespace cudaCompress;


#include "../cudaCompress/src/examples/tthread/tinythread.h"

#include "../cudaCompress/src/examples/tools/entropy.h"
#include "../cudaCompress/src/examples/tools/imgtools.h"
#include "../cudaCompress/src/examples/tools/rawfile.h"

#include "../cudaCompress/src/examples/cudaUtil.h"

#include "../cudaCompress/src/examples/CompressImage.h"
#include "../cudaCompress/src/examples/CompressHeightfield.h"
#include "../cudaCompress/src/examples/CompressVolume.h"

#include "../Timer/Timer.h"



float * decompress(int3 size, uint * h_data, const float & Quant_step, GPUResources::Config & config, GPUResources & shared, CompressVolumeResources & res, size_t & bufferSize)
{

	// Size of the compress file
	const unsigned int elemCountTotal = size.x*size.y*size.z;

	float * dp_field = nullptr;

	// Allocate GPU Memory
	gpuErrchk(cudaMalloc(&dp_field, elemCountTotal * sizeof(float)));


	const bool doRLEOnlyOnLvl0 = true;

	// Pin memory for the copy operation
	gpuErrchk(cudaHostRegister(h_data, bufferSize, cudaHostRegisterDefault));

	Timer timer;

	// Decompress the field
	TIMELAPSE(decompressVolumeFloat(shared, res, dp_field, size.x, size.y, size.z, 2, h_data, bufferSize*8, 0.01f,doRLEOnlyOnLvl0), "Decompression Function");

	// unpin host memory
	gpuErrchk(cudaHostUnregister(h_data));

	// return device pointer to the decompressed field 
	return dp_field;
}

void releaseGPUResources(float * dp_field)
{
	cudaFree(dp_field);
}


void DecompressResources::initializeDecompressionResources(int3 size)
{
	this->config = CompressVolumeResources::getRequiredResources(size.x, size.y, size.z, 1, huffmanBits);
	this->shared.create(config);
	this->res.create(shared.getConfig());
}

void DecompressResources::releaseDecompressionResources()
{
	res.destroy();
	shared.destroy();
}