#pragma once
#include <string>
#include "..//Options/SolverOptions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Particle/Particle.h"
#include "..//Volume/Volume_IO.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"
#include "..//Raycaster/Raycasting.h"

class DispersionTracer : public Raycasting
{
public:

	bool initialize() override;

	
	__host__ void setResources 
	(
		Camera* _camera,
		int* _width,
		int* _height,
		SolverOptions* _solverOption,
		RaycastingOptions* _raycastingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext,
		DispersionOptions* _dispersionOptions
	);

	bool release() override;
	void rendering() override;
	bool updateScene() override;
	void trace();
	void retrace();


private:

	bool InitializeParticles();
	bool InitializeHeightArray();
	bool InitializeHeightSurface();
	bool InitializeHeightTexture();
	DispersionOptions* dispersionOptions;


	Particle* h_particle = nullptr;
	Particle* d_particle = nullptr;
	
	unsigned int n_particles;
	
	CudaSurface heightSurface;			// cuda surface storing the results
	cudaTextureObject_t heightFieldTexture;
	CudaArray_2D<float4> heightArray;
		
};



