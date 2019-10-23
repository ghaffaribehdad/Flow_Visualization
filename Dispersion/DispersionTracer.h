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

	bool initialize
	(
		cudaTextureAddressMode addressMode_X ,
		cudaTextureAddressMode addressMode_Y ,
		cudaTextureAddressMode addressMode_Z 
	) override;

	
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
	bool retrace();
	void gradient();


private:

	bool InitializeParticles();
	bool InitializeHeightArray();
	bool InitializeHeightArray3D();
	bool InitializeHeight_gradient_Array();

	bool InitializeHeightSurface();
	bool InitializeHeight_gradient_Surface();

	bool InitializeHeight_gradient_Texture();

	DispersionOptions* dispersionOptions;


	Particle* h_particle = nullptr;
	Particle* d_particle = nullptr;
	
	unsigned int n_particles;
	
	CudaSurface heightSurface;			// cuda surface storing the results
	CudaSurface heightSurface_gradient;
	cudaTextureObject_t heightFieldTexture;
	CudaArray_2D<float4> heightArray_gradient;
	CudaArray_2D<float> heightArray;
	CudaArray_3D<float4> heightArray3D;
		
};



