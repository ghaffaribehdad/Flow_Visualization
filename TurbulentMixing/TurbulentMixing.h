#pragma once
#include "..//Raycaster/Raycasting.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Volume/Volume_IO_X_Major.h"
#include "../Options/SolverOptions.h"
#include "../Options/TurbulentMixingOptions.h"

class TurbulentMixing : public Raycasting
{

private:
	
	CudaArray_2D<float4>	a_mixing; // To store turbulence kinetic energy
	CudaSurface				s_mixing; // Bind to "a_mixing" array

	VolumeTexture2D			v_field_t0; // Volumetric velocity field at t0
	//VolumeTexture2D			v_field_t1; // Volumetric velocity field at t1

	Volume_IO_X_Major		volumeIO; // To read the volumetric data from file

	TurbulentMixingOptions*		turbulentMixingOptions;

private:

	bool updateVolume(VolumeTexture2D& v_field, int & idx);		// Updates volumetric velocity fields
	
	void dissipate();	// Dissipate	Momentum
	void create();		// Create		Momentum
	void advect();		// Advect		Momentum 

public:
	bool initalize();
	
	void update(int timestep);

	bool release();

	virtual void rendering() override;

	virtual void show(RenderImGuiOptions* renderImGuiOptions) override
	{
		//if (this->renderImGuiOptions.showTurbulentMixing)
	//{
	//	if (!this->turbulentMixingOptions.initialized)
	//	{


	//		this->turbulentMixingOptions.initialized  = this->turbulentMixing.initalize();
	//	}

	//	this->turbulentMixing.draw();

	//}

	//if (this->renderImGuiOptions.updateTurbulentMixing)
	//{
	//	this->turbulentMixing.updateScene();

	//	renderImGuiOptions.updateTurbulentMixing = false;
	//}

	//if (this->renderImGuiOptions.releaseTurbulentMixing)
	//{
	//	this->turbulentMixing.release();
	//	this->renderImGuiOptions.releaseTurbulentMixing = false;
	//}
	}


	void setResources
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
		TurbulentMixingOptions* _turbulentMixingOptions
	)
	{
		Raycasting::setResources
		(
			this->camera = _camera,
			this->width = _width,
			this->height = _height,
			this->solverOptions = _solverOption,
			this->raycastingOptions = _raycastingOptions,
			this->renderingOptions = _renderingOptions,
			this->device = _device,
			this->pAdapter = _pAdapter,
			this->deviceContext = _deviceContext
		);
		this->turbulentMixingOptions = _turbulentMixingOptions;
	}



};