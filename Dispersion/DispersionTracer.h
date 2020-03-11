#pragma once
#include <string>
#include "..//Options/SolverOptions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Particle/Particle.h"
#include "..//VolumeIO/Volume_IO.h"
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


	virtual void show(RenderImGuiOptions* renderImGuiOptions) override
	{
		if (renderImGuiOptions->showDispersion)
		{
			if (this->dispersionOptions->retrace)
			{
				this->retrace();
				this->dispersionOptions->retrace = false;
			}
			if (!this->dispersionOptions->initialized)
			{
				this->initialize(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				this->dispersionOptions->initialized = true;
			}
			this->draw();

			if (renderImGuiOptions->updateDispersion)
			{
				this->updateScene();
				renderImGuiOptions->updateDispersion = false;

			}
		}
		else
		{
			if (dispersionOptions->released)
			{
				this->release();
				dispersionOptions->released = false;
			}
		}
	}

	bool retrace();
	virtual void gradient3D_Single();
	void gradient3D_Double();

protected:

	VolumeTexture3D t_velocityField_0;
	VolumeTexture3D t_velocityField_1;


	virtual void trace3D_path_Single();
	void trace3D_path_Double();


	virtual bool initializeShaders() override;

	Particle* h_particle = nullptr;
	Particle* d_particle = nullptr;


	unsigned int n_particles;


	virtual bool InitializeParticles();
	virtual bool singleSurfaceInitialization();
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



