#pragma once
#include "cuda_runtime.h"
#include <vector>


typedef unsigned int uint;

float * decompress(int3 size, std::vector<uint> & h_data, const float & Quant_step);
void releaseGPUResources(float * dp_field);