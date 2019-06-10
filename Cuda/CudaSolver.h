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
#include "../Volume/Volume_IO.h"

template <class T1>
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

	// Solver Parameters
	SolverOptions solverOptions;

protected:

	cudaGraphicsResource* cudaGraphics;

	// A COM pointer to the vector Field
	bool InitializeCUDA();
	bool InitializeVolumeIO();
	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* adapter;
	ID3D11Device* device;

	Volume_IO volume_IO;


	void* p_VertexBuffer;
};