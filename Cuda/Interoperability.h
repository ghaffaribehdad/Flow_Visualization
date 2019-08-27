#pragma once
#include "cuda_runtime.h"
#include <d3d11.h>
#include "../ErrorLogger.h"

struct Interoperability_desc
{
	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* p_adapter = nullptr;
	ID3D11Device* p_device = nullptr;
	size_t size;
	cudaGraphicsRegisterFlags flag = cudaGraphicsRegisterFlagsNone;
	ID3D11Resource * pD3DResource;
};

class Interoperability
{

private:

	cudaGraphicsResource_t cudaGraphics = NULL;
	Interoperability_desc interoperability_desc;
	

public:

	void setInteroperability_desc(
		cudaDeviceProp _cuda_device_prop,
		IDXGIAdapter* _adapter,
		ID3D11Device* _device,
		ID3D11Resource* _pD3DResource,
		size_t _size,
		cudaGraphicsRegisterFlags _flag
	)
	{
		interoperability_desc.cuda_device_prop = _cuda_device_prop;
		interoperability_desc.p_adapter = _adapter;
		interoperability_desc.p_device = _device;
		interoperability_desc.pD3DResource = _pD3DResource;
		interoperability_desc.size = _size;
		interoperability_desc.flag = _flag;
	}

	void setInteroperability_desc(const Interoperability_desc & _interoperability_desc)
	{
		interoperability_desc = _interoperability_desc;
	}


	bool InitializeResource()
	{

		// Register DX Resource to map it
		gpuErrchk(cudaGraphicsD3D11RegisterResource(
			&this->cudaGraphics,
			this->interoperability_desc.pD3DResource,
			this->interoperability_desc.flag));

		// Map graphic resoource
		gpuErrchk(cudaGraphicsMapResources(
			1,
			&this->cudaGraphics
		));

		return true;
	}
	

	void release()
	{
		gpuErrchk(cudaGraphicsUnmapResources(1, &this->cudaGraphics));
		gpuErrchk(cudaGraphicsUnregisterResource(this->cudaGraphics));
	}

	void getMappedArray(cudaArray_t & cuda_Array)
	{
		gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&cuda_Array, this->cudaGraphics, 0, 0));
	}

	void getMappedPointer(void * p_resource)
	{
		gpuErrchk(cudaGraphicsResourceGetMappedPointer(
			&p_resource,
			&interoperability_desc.size,
			this->cudaGraphics
		));
	}

	
};

