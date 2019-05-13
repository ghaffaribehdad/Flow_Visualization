#pragma once


#include "StringConverter.h"
#include <Windows.h>
#include <cuda_runtime_api.h>


class ErrorLogger
{
public:
	static void Log(std::string message);
	static void Log(HRESULT hr, std::string message);
	static void Log(HRESULT hr, std::wstring message);
	static void Log(cudaError_t code, const char *file, int line, bool abort = true);

}; 


