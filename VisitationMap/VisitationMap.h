#pragma once

#include "../Raycaster/Raycasting.h"
#include "../Cuda/CudaArray.h"
#include "../Cuda/cudaSurface.h"
#include "../Options/VisitationOptions.h"

class VisitationMap : public Raycasting
{

public:

	bool initialization();
	virtual void show(RenderImGuiOptions* renderImGuiOptions) override;
	virtual void rendering() override;

	void setResources(Camera* _camera,
		int* _width,
		int* _height,
		SolverOptions* _solverOption,
		RaycastingOptions* _raycastingOptions,
		RenderingOptions* _renderingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext,
		VisitationOptions* _visitationOptions
	) 
	{

		this->camera = _camera;
		this->FOV_deg = 30.0;
		this->width = _width;
		this->height = _height;

		this->visitationOptions = _visitationOptions;
		this->solverOptions = _solverOption;
		this->raycastingOptions = _raycastingOptions;
		this->renderingOptions = _renderingOptions;

		this->device = _device;
		this->pAdapter = _pAdapter;
		this->deviceContext = _deviceContext;
	}



	bool reinitialization();
private:

	VisitationOptions * visitationOptions;
	bool copyToHost();
	bool writeToFile();
	bool generateVisitationMap();
	CudaSurface s_visitationMap;
	CudaArray_3D<float4> a_visitationMap;
	void updateEnsembleMember();
};