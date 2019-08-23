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
#include "..//Enum.h"

class CUDASolver
{
public:
	CUDASolver();


	bool Initialize(SolverOptions * _solverOptions);

	// Read volume and store it in a vector, returns a pointer to the first element of the vector
	__host__ float* InitializeVelocityField(int ID);

	// Solve must be defined in the derived classes
	virtual bool solve()
	{
		return true;
	}

	bool FinalizeCUDA();

	// Solver Parameters
	SolverOptions * solverOptions;

protected:
	
	__host__ void InitializeParticles(SeedingPattern seedingPattern);

	cudaGraphicsResource* cudaGraphics = NULL;

	// A COM pointer to the vector Field
	bool InitializeCUDA();

	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* adapter = NULL;
	ID3D11Device* device = NULL;

	Volume_IO volume_IO;

	// The probe particles
	Particle<float>* d_Particles = nullptr;
	Particle<float>* h_Particles = nullptr;

	virtual void release(){}

	void* p_VertexBuffer = NULL;
};