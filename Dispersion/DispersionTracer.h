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
	void trace3D();	 //Traces the streamlines


	void trace3D_path_Single();
	void trace3D_path_Double();


	bool retrace();
	void gradient3D_Single();
	void gradient3D_Double();

private:

	VolumeTexture3D velocityField_0;
	VolumeTexture3D velocityField_1;


	virtual bool initializeShaders() override;

	Particle* h_particle = nullptr;
	Particle* d_particle = nullptr;

	unsigned int n_particles;

	bool InitializeParticles();
protected:


	bool singleSurfaceInitialization();
	bool doubleSurfaceInitialization();

	bool InitializeHeightArray3D_Single(int x, int y ,int z);
	virtual bool InitializeHeightArray3D_Single(int3 gridSize);

	bool InitializeHeightArray3D_Double(int x, int y, int z);
	virtual bool InitializeHeightArray3D_Double(int3 gridSize);


	virtual bool InitializeHeightSurface3D_Single();
	virtual bool InitializeHeightSurface3D_Double();


	virtual bool InitializeHeightTexture3D_Single();
	virtual bool InitializeHeightTexture3D_Double();


	bool LoadVelocityfield(const unsigned int & idx);

	DispersionOptions* dispersionOptions;

	// Primary Height Surface
	CudaSurface s_HeightSurface_Primary;
	CudaSurface s_HeightSurface_Primary_Ex;

	cudaTextureObject_t t_HeightSurface_Primary;
	cudaTextureObject_t t_HeightSurface_Primary_Ex;

	CudaArray_3D<float4> a_HeightSurface_Primary;
	CudaArray_3D<float4> a_HeightSurface_Primary_Ex;		


	// Secondary Height Surface
	CudaSurface s_HeightSurface_Secondary;
	CudaSurface s_HeightSurface_Secondary_Ex;

	cudaTextureObject_t t_HeightSurface_Secondary;
	cudaTextureObject_t t_HeightSurface_Secondary_Ex;

	CudaArray_3D<float4> a_HeightSurface_Secondary;
	CudaArray_3D<float4> a_HeightSurface_Secondary_Ex;
};



