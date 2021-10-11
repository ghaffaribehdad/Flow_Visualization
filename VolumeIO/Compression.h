#pragma once
#include "cuda_runtime.h"
#include <vector>
#include "../cudaCompress/src/examples/CompressVolume.h"
#include "../Options/SolverOptions.h"
#include "../Options/FieldOptions.h"

typedef unsigned int uint;

float * decompress(int3 size, uint * h_data, const float & Quant_step, GPUResources::Config & config, GPUResources & shared, CompressVolumeResources & res, size_t & bufferSize);
void releaseGPUResources(float * dp_field);


struct DecompressResources
{

private:
	int3 gridSize;
	size_t maxSize = 0;
	float * dp_field = nullptr;
	unsigned int * pHost = nullptr;
	void allocateAndRegister();

public:
	GPUResources::Config config;
	GPUResources shared;
	CompressVolumeResources res;
	uint huffmanBits = 0;
	void initializeDecompressionResources(SolverOptions * solverOption, unsigned int * _pHost);
	void initializeDecompressionResources(FieldOptions * fieldOptions, unsigned int * _pHost);
	void initializeDecompressionResources(std::size_t & _maxSize, int * _gridSize, unsigned int * _pHost);
	void releaseDecompressionResources();
	void pinHostMemory(size_t & maxSize);
	void unpinHostMemory();

	
	void decompress(uint * h_data, const float & Quant_step, size_t & bufferSize);
	void releaseGPUResources();


public:


	float * getDevicePointer()
	{
		return dp_field;
	}



};