#pragma once

#include <d3d11.h>
#include "CudaDevice.h"
#include <wrl/client.h>
#include "../BinaryStream/BinaryReader.h"
#include "../BinaryStream/BinaryWriter.h"
#include "../ErrorLogger.h"
#include <DirectXMath.h>
#include "../Particle.cuh"
#include "../VelocityField.cuh"
#include "../SolverOptions.h"
#include "CudaDevice.h"
#include "../Graphics/Vertex.h"

class CUDASolver
{
public:
	CUDASolver();


	bool Initialize(SolverOptions _solverOptions);

	// Solve must be defined in the derived classes
	virtual bool solve()
	{
		return true;
	}

	bool FinalizeCUDA();

protected:
	
	// Read and copy file into memeory and save a pointer to it
	bool ReadField(std::vector<char>* p_vec_buffer, std::string fileName);

	// Upload Field to GPU and returns a pointer to the data on GPU
	template <class T>
	T * UploadToGPU(T * host_Data, size_t _size)
	{
		T* device_data;

		gpuErrchk(cudaMalloc((void**)& device_data, _size));

		gpuErrchk(cudaMemcpy(device_data, host_Data, _size, cudaMemcpyHostToDevice));

		return  device_data;
	}



	// Particle tracing parameters
	Particle* h_particles;
	// Solver Parameters
	SolverOptions solverOptions;
	cudaGraphicsResource* cudaGraphics;

	// A COM pointer to the vector Field
	bool InitializeCUDA();
	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* adapter;
	ID3D11Device* device;

	void* p_VertexBuffer;
};