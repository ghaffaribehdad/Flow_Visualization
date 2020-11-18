#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "../Graphics/Vertex.h"
#include "../VolumeIO/Volume_IO_Z_Major.h"
#include "../VolumeTexture/VolumeTexture.h"




class CUDASolver
{
public:
	CUDASolver();


	bool Initialize(SolverOptions * _solverOptions);

	bool Reinitialize();

	// Solve must be defined in the derived classes
	virtual bool solve()
	{
		return true;
	}
	__host__ virtual bool initializeRealtime(SolverOptions * p_solverOptions) { return true; };
	__host__ virtual bool resetRealtime() { return true; };
	bool FinalizeCUDA();

	// Solver Parameters
	SolverOptions * solverOptions;

protected:
	
	void releaseVolumeIO();
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

	virtual bool release() = 0;

	void* p_VertexBuffer = NULL;

	void loadTexture
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX = cudaAddressModeWrap,
		cudaTextureAddressMode addressModeY = cudaAddressModeBorder,
		cudaTextureAddressMode addressModeZ = cudaAddressModeWrap
	);


	void loadTextureCompressed
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX = cudaAddressModeWrap,
		cudaTextureAddressMode addressModeY = cudaAddressModeBorder,
		cudaTextureAddressMode addressModeZ = cudaAddressModeWrap
	);
};