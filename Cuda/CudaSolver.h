#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "../Particle/Particle.h"
#include "../Options/SolverOptions.h"
#include "../Graphics/Vertex.h"
#include "../VolumeIO/Volume_IO_Z_Major.h"
#include "../VolumeTexture/VolumeTexture.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"



class CUDASolver
{
public:
	CUDASolver();

	SolverOptions * solverOptions;


	virtual bool initializeRealtime(SolverOptions * p_solverOptions) { return true; };
	virtual bool resetRealtime() { return true; };
	virtual bool solve() { return true; }

	bool Initialize(SolverOptions * _solverOptions);
	bool Reinitialize();
	bool FinalizeCUDA();

	__host__ bool checkFile(SolverOptions * p_solverOptions)
	{
		return volume_IO.file_check(p_solverOptions->filePath + p_solverOptions->fileName + std::to_string(p_solverOptions->currentIdx) + ".bin");
	}

protected:
	

	// CUDA interoperation resources
	cudaGraphicsResource* cudaGraphics = NULL;
	void* p_VertexBuffer = NULL;
	cudaDeviceProp cuda_device_prop;
	IDXGIAdapter* adapter = NULL;
	ID3D11Device* device = NULL;


	// Particle Tracing Resources
	Particle* d_Particles = nullptr;
	Volume_IO_Z_Major volume_IO;
	VolumeTexture3D volumeTexture;

	bool InitializeCUDA();
	virtual bool release() = 0;
	void releaseVolumeIO();
	void initializeParticles(SeedingPattern seedingPattern);
	void loadTexture
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX = cudaAddressModeWrap,
		cudaTextureAddressMode addressModeY = cudaAddressModeBorder,
		cudaTextureAddressMode addressModeZ = cudaAddressModeWrap,
		bool normalized = false
	);

	void loadTextureCompressed
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX = cudaAddressModeWrap,
		cudaTextureAddressMode addressModeY = cudaAddressModeBorder,
		cudaTextureAddressMode addressModeZ = cudaAddressModeWrap,
		bool normalized = false
	);



};