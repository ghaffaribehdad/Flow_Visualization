#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "../Graphics/Vertex.h"
#include "../VolumeIO/Volume_IO_Z_Major.h"




class CUDASolver
{
public:
	CUDASolver();


	bool Initialize(SolverOptions * _solverOptions);

	

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

	Volume_IO_Z_Major volume_IO;

	// The probe particles
	Particle* d_Particles = nullptr;
	Particle* h_Particles = nullptr;

	virtual void release(){}

	void* p_VertexBuffer = NULL;
};