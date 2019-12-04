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


class HeightfieldGenerator : public Raycasting
{
public:

	virtual bool initialize
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
		RenderingOptions* _renderingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext,
		DispersionOptions* _dispersionOptions
	);

	bool release() override;
	void rendering() override;
	virtual bool updateScene() override;
	void trace3D();
	void trace3D_path();
	bool retrace();
	void gradient3D();

private:

	VolumeTexture3D velocityField_0;
	VolumeTexture3D velocityField_1;

	Particle* h_particle = nullptr;
	Particle* d_particle = nullptr;

	unsigned int n_particles;

	bool InitializeParticles();
protected:




	bool InitializeHeightArray3D(int x, int y ,int z);
	virtual bool InitializeHeightArray3D(int3 gridSize);
	virtual bool InitializeHeightSurface3D();
	virtual bool InitializeHeightTexture3D();
	bool LoadVelocityfield(const unsigned int & idx);

	DispersionOptions* dispersionOptions;


	CudaSurface heightSurface3D;
	CudaSurface heightSurface3D_extra;

	cudaTextureObject_t heightFieldTexture3D;

	cudaTextureObject_t heightFieldTexture3D_extra;

	CudaArray_3D<float4> heightArray3D;
	CudaArray_3D<float4> heightArray3D_extra;		
};



