#pragma once

#include "Heightfield.h"
#include "..//Options/fluctuationheightfieldOptions.h"
#include "..//Raycaster/BoundingBox.h"

struct size_t3
{
	size_t x;
	size_t y;
	size_t z;
};

class FluctuationHeightfield : public Heightfield
{
	// 
public:
	bool initialize
	(
		cudaTextureAddressMode addressMode_X,
		cudaTextureAddressMode addressMode_Y,
		cudaTextureAddressMode addressMode_Z
	) override;


	void setResources(
		Camera* _camera,
		int* _width,
		int* _height,
		SolverOptions* _solverOption,
		RaycastingOptions* _raycastingOptions,
		RenderingOptions* _renderingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext,
		DispersionOptions* _dispersionOptions,
		TimeSpaceRenderingOptions* timeSpaceRenderingOptions
	)
	{
		this->camera = _camera;
		this->FOV_deg = 30.0;
		this->width = _width;
		this->height = _height;

		this->solverOptions = _solverOption;
		this->raycastingOptions = _raycastingOptions;
		this->renderingOptions = _renderingOptions;

		this->device = _device;
		this->pAdapter = _pAdapter;
		this->deviceContext = _deviceContext;
		this->dispersionOptions = _dispersionOptions;
		this->timeSpaceRenderingOptions = timeSpaceRenderingOptions;
	}

	virtual void show(RenderImGuiOptions* renderImGuiOptions) override
	{
		if (renderImGuiOptions->showFluctuationHeightfield)
		{

			if (!this->timeSpaceRenderingOptions->initialized)
			{

				this->initialize(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
				this->timeSpaceRenderingOptions->initialized = true;
			}
			// Overrided draw
			this->draw();

			if (renderImGuiOptions->updatefluctuation)
			{
				this->updateScene();

				renderImGuiOptions->updatefluctuation = false;

			}
		}
		else
		{

		}
	}

	void generateTimeSpaceField3D(TimeSpaceRenderingOptions * timeSpaceOptions);
	

	void gaussianFilter();
	virtual void rendering() override;
	__host__ bool initializeBoundingBox() override;


	TimeSpaceRenderingOptions* timeSpaceRenderingOptions;
private:

	int3 m_gridSize3D = { 0,0,0 };
	int2 m_gridSize2D = { 0,0 };
	CudaArray_2D<float> heightArray2D;
	cudaTextureObject_t heightFieldTexture2D;


	virtual bool InitializeHeightSurface3D(cudaArray_t cudaArray);


};





