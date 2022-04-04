#include "Compression.h"
#include "../ErrorLogger/ErrorLogger.h"


#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../Cuda/helper_math.h"
#include "../Options/SolverOptions.h"
#include "../Options/FieldOptions.h"
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cudaCompressInclude/Instance.h>
#include <cudaCompressInclude/Encode.h>
#include <cudaCompressInclude/util/Bits.h>
#include <cudaCompressInclude/util/DWT.h>
#include <cudaCompressInclude/util/Quantize.h>
#include <cudaCompressInclude/util/YCoCg.h>
#include <cudaCompressInclude/Timing.h>
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

	Timer timer;

	
	// Size of the compress file
	const unsigned int elemCountTotal = size.x*size.y*size.z;

	float * dp_field = nullptr;
	TIMELAPSE(
	// Allocate GPU Memory
	gpuErrchk(cudaMalloc(&dp_field, elemCountTotal * sizeof(float)));
	, "Decompression Function Part1, register host memory");

	const bool doRLEOnlyOnLvl0 = true;
	TIMELAPSE(
	// Pin memory for the copy operation
	gpuErrchk(cudaHostRegister(h_data, bufferSize, cudaHostRegisterDefault));
	, "Pinning the memory");
	

	TIMELAPSE(
	// Decompress the field
	decompressVolumeFloat(shared, res, dp_field, size.x, size.y, size.z, 2, h_data, bufferSize*8, 0.01f,doRLEOnlyOnLvl0);
	, "Decompression Function Part2, is the decompression and cannot be optimized");

	TIMELAPSE(
	// unpin host memory
	gpuErrchk(cudaHostUnregister(h_data));
	, "Decompression Function Part3, unregister host memory");
	

	// return device pointer to the decompressed field 
	return dp_field;
}

void releaseGPUResources(float * dp_field)
{
	cudaFree(dp_field);
}


void DecompressResources::releaseGPUResources()
{
	cudaFree(dp_field);
}


void DecompressResources::initializeDecompressionResources(SolverOptions * solverOption, unsigned int * _pHost)
{
	gridSize = Array2Int3(solverOption->gridSize);
	//gridSize.y = gridSize.y * 4; // Since there are 4 channels;
	gridSize.x = gridSize.x * 4; // Since there are 4 channels;
	

	this->config = CompressVolumeResources::getRequiredResources(gridSize.x, gridSize.y, gridSize.z, 1, huffmanBits);
	this->shared.create(config);
	this->res.create(shared.getConfig());


	//this->allocateAndRegister();
	this->pHost = _pHost;
	this->pinHostMemory(solverOption->maxSize);
}


void DecompressResources::initializeDecompressionResources(FieldOptions * fieldOptions, unsigned int * _pHost)
{
	gridSize = Array2Int3(fieldOptions->gridSize);
	//gridSize.y = gridSize.y * 4; // Since there are 4 channels;
	gridSize.x = gridSize.x * 4; // Since there are 4 channels;


	this->config = CompressVolumeResources::getRequiredResources(gridSize.x, gridSize.y, gridSize.z, 1, huffmanBits);
	this->shared.create(config);
	this->res.create(shared.getConfig());


	//this->allocateAndRegister();
	this->pHost = _pHost;
	this->pinHostMemory(fieldOptions->fileSizeMaxByte);
}



void DecompressResources::initializeDecompressionResources(std::size_t & _maxSize, int * _gridSize, unsigned int * _pHost)
{
	gridSize = Array2Int3(_gridSize);
	//gridSize.y = gridSize.y * 4; // Since there are 4 channels;
	gridSize.x = gridSize.x * 4; // Since there are 4 channels;



	this->config = CompressVolumeResources::getRequiredResources(gridSize.x, gridSize.y, gridSize.z, 1, huffmanBits);
	this->shared.create(config);
	this->res.create(shared.getConfig());


	//this->allocateAndRegister();
	this->pHost = _pHost;
	this->pinHostMemory(_maxSize);
}

void DecompressResources::releaseDecompressionResources()
{
	res.destroy();
	shared.destroy();
	unpinHostMemory();
}



void DecompressResources::decompress(uint * h_data, const float & Quant_step, size_t & bufferSize)
{
	 
	// Decompress the field
	// Pin memory for the copy operation

	cudaFree(dp_field);
	const unsigned int elemCountTotal = gridSize.x*gridSize.y*gridSize.z;
	// Allocate GPU Memory
	gpuErrchk(cudaMalloc(&dp_field, elemCountTotal * sizeof(float)));

	decompressVolumeFloat(shared, res, dp_field, gridSize.x, gridSize.y, gridSize.z, 2, pHost, bufferSize * 8, 0.01f, true);


}


void DecompressResources::allocateAndRegister()
{
	// Size of the compress file
	const unsigned int elemCountTotal = gridSize.x*gridSize.y*gridSize.z;
	// Allocate GPU Memory
	gpuErrchk(cudaMalloc(&dp_field, elemCountTotal * sizeof(float)));
}


void DecompressResources::pinHostMemory(size_t & maxSize)
{

	gpuErrchk(cudaHostRegister(pHost, maxSize, cudaHostRegisterDefault));
}

void DecompressResources::unpinHostMemory()
{
	gpuErrchk(cudaHostUnregister(pHost));
}