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

	// Read volume and store it in a vector, returns a pointer to the first element of the vector
	__host__ T1* InitializeVelocityField(int ID);

	// Solve must be defined in the derived classes
	virtual bool solve()
	{
		return true;
	}

	bool FinalizeCUDA();

	// Solver Parameters
	SolverOptions solverOptions;

protected:

	__host__ bool InitializeTexture(T1* h_source, cudaTextureObject_t& texture);
	
	__host__ void InitializeParticles();

	cudaGraphicsResource* cudaGraphics = NULL;

	// A COM pointer to the vector Field
	bool InitializeCUDA();

	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* adapter = NULL;
	ID3D11Device* device = NULL;

	Volume_IO volume_IO;

	// The probe particles
	Particle<T1>* d_Particles;
	Particle<T1>* h_Particles;

	void* p_VertexBuffer = NULL;
};