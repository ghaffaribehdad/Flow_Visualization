#pragma once
#include "cuda_runtime.h"
#include <d3d11.h>
#include "../ErrorLogger.h"

struct Interoperability_desc
{
	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* p_adapter = NULL;
	ID3D11Device* p_device = NULL;
	ID3D11Resource* pD3DResource = NULL;
	size_t size;
};

class Interoperability
{

private:

	Interoperability_desc interoperability_desc;
	cudaGraphicsResource* cudaGraphics = NULL;

public:

	void setInteroperability_desc(
		cudaDeviceProp _cuda_device_prop,
		IDXGIAdapter* _adapter,
		ID3D11Device* _device,
		ID3D11Resource* _pD3DResource,
		size_t _size)
	{
		interoperability_desc.cuda_device_prop = _cuda_device_prop;
		interoperability_desc.p_adapter = _adapter;
		interoperability_desc.p_device = _device;
		interoperability_desc.pD3DResource = _pD3DResource;
		interoperability_desc.size = _size;
	}

	void setInteroperability_desc(Interoperability_desc& _interoperability_desc)
	{
		interoperability_desc = _interoperability_desc;
	}


	bool InitializeResource()
	{
		// Get number of CUDA-Enable devices
		int device;
		gpuErrchk(cudaD3D11GetDevice(&device, solverOptions.this->adapter));

		// Get properties of the Best(usually at slot 0) card
		gpuErrchk(cudaGetDeviceProperties(&this->cuda_device_prop, 0));

		// Register Vertex Buffer to map it
		gpuErrchk(cudaGraphicsD3D11RegisterResource(
			&this->cudaGraphics,
			this->interoperability_desc.pD3DResource,
			cudaGraphicsRegisterFlagsNone));

		// Map Vertex Buffer
		gpuErrchk(cudaGraphicsMapResources(
			1,
			&this->cudaGraphics
		));

		// Get Mapped pointer
		size_t size = //size of the resource

		gpuErrchk(cudaGraphicsResourceGetMappedPointer(
			&p_VertexBuffer,
			&size,
			this->cudaGraphics
		));

		return true;
	}
	

	void release();
};
