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



float * decompress(int3 size, std::vector<uint> & h_data, const float & Quant_step)
{


	std::vector<uint> bitStream(h_data.begin(), h_data.end());
	// Size of the compress file
	const unsigned int elemCountTotal = size.x*size.y*size.z;

	float * dp_field = nullptr;

	// Allocate GPU Memory
	gpuErrchk(cudaMalloc(&dp_field, elemCountTotal * sizeof(float)));
	// Set it to zeros
	gpuErrchk(cudaMemset(dp_field, 0, elemCountTotal * sizeof(float)));

	const bool doRLEOnlyOnLvl0 = true;


	uint huffmanBits = 0;


	GPUResources::Config config = CompressVolumeResources::getRequiredResources(size.x, size.y, size.z, 1, huffmanBits);
	GPUResources shared;

	shared.create(config);
	CompressVolumeResources res;
	res.create(shared.getConfig());

	cudaSafeCall(cudaHostRegister(bitStream.data(), bitStream.size() * sizeof(uint), cudaHostRegisterDefault));


	decompressVolumeFloat(shared, res, dp_field, size.x, size.y, size.z, 2, bitStream, 0.01f, doRLEOnlyOnLvl0);
	cudaSafeCall(cudaHostUnregister(bitStream.data()));
	cudaSafeCall(cudaFreeHost(bitStream.data()));
	res.destroy();
	shared.destroy();

	return dp_field;
}

void releaseGPUResources(float * dp_field)
{
	cudaFree(dp_field);
}