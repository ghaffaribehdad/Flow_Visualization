#pragma once


#include "StringConverter.h"
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>


// Create Error Logging for GPU
#define gpuErrchk(ans) { ErrorLogger::Log((ans), __FILE__, __LINE__); }

class ErrorLogger
{
public:
	static void Log(std::string message);
	static void Log(HRESULT hr, std::string message);
	static void Log(HRESULT hr, std::wstring message);
	static void Log(cudaError_t code, const char *file, int line, bool abort = true);

}; 


