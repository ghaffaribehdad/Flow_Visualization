#pragma once

#include "FieldGenerator3D.h"
#include "../Options/TimeSpace3DOptions.h"


class TimeSpaceField : public FieldGenerator3D
{
public:


	virtual void show(RenderImGuiOptions* renderImGuiOptions) override;


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
		TimeSpace3DOptions* _timeSpace3DOptions
	);

private:

	virtual bool cudaRaycaster() override;

	TimeSpace3DOptions * timeSpace3DOptions;

	virtual bool generateVolumetricField
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	) override;
	virtual bool regenerateVolumetricField() override;

};



__global__ void  generateVorticityFieldSpaceTime
(
	cudaSurfaceObject_t s_vorticity,
	cudaTextureObject_t t_velocityField,
	SolverOptions solverOptions,
	TimeSpace3DOptions timeSpace3DOptions,
	int timestep,
	int3 gridSize
);


