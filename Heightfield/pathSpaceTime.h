#pragma once
#include "Heightfield.h"
#include "..//Options/pathSpaceTimeOptions.h"
#include "..//Raycaster/BoundingBox.h"


class PathSpaceTime : public Heightfield
{
	// 
public:
	bool initialize
	(
		cudaTextureAddressMode addressMode_X,
		cudaTextureAddressMode addressMode_Y,
		cudaTextureAddressMode addressMode_Z
	) override;


	__host__ virtual bool updateconstantBuffer() override
	{
		PS_constantBuffer.data.transparency = 0;
		PS_constantBuffer.ApplyChanges();

		return true;
	}

	bool initializeRaycasting()
	{
		if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
			return false;


		if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
			return false;

		// set the number of rays = number of pixels
		this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

		return false;
	}

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
		PathSpaceTimeOptions* _pathSpaceTimeOptions,
		FieldOptions * _fieldOptions
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
		this->pathSpaceTimeOptions = _pathSpaceTimeOptions;
		this->fieldOptions = _fieldOptions;
	}

	virtual void show(RenderImGuiOptions* renderImGuiOptions) override
	{
		if (renderImGuiOptions->showPathSpaceTime)
		{

			if (!this->pathSpaceTimeOptions->initialized)
			{

				this->initialize(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
				this->pathSpaceTimeOptions->initialized = true;
			}

			// Overrided draw
			this->draw();

			if (renderImGuiOptions->updatePathSpaceTime || renderImGuiOptions->updateRaycasting)
			{
				this->updateScene();

				renderImGuiOptions->updatefluctuation = false;

			}
		}
		else
		{

		}
	}

	void generatePathField3D();
	bool InitializePathSurface();
	virtual void rendering() override;
	PathSpaceTimeOptions* pathSpaceTimeOptions;

private:

	int3 m_gridSize3D = { 0,0,0 };
	int2 m_gridSize2D = { 0,0 };
	CudaArray_2D<float> heightArray2D;

	//cudaTextureObject_t heightFieldTexture2D;
	CudaSurface s_PathSpaceTime;
	CudaArray_3D<float4> a_PathSpaceTime;

	//virtual bool initializeBoundingBox() override;
	//virtual bool InitializeHeightSurface3D(cudaArray_t cudaArray);
	//bool loadRaycastingTexture(FieldOptions * fieldOptions, int idx);


};