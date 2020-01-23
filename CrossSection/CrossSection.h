#pragma once

#include "../Raycaster/Raycasting.h"
#include "../Cuda/CudaArray.h"
#include "../Options/CrossSectionOptions.h"
#include "../VolumeTexture/VolumeTexture.h"
#include "../Cuda/cudaSurface.h"

class CrossSection : public Raycasting
{

private:

	CudaSurface				s_filteringSurface;
	CudaArray_3D<float4>	a_field;

	int3 m_dimension = { 0,0,0 };

	//Options
	CrossSectionOptions* crossSectionOptions;
	VolumeTexture3D t_volumeTexture;
	VolumeTexture3D t_volumeGradient;


	void filterExtermum();
	void initializedFilterSurface();

public:

	virtual bool initialize
	(
		cudaTextureAddressMode addressMode_X,
		cudaTextureAddressMode addressMode_Y,
		cudaTextureAddressMode addressMode_Z
	)  override;

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
		CrossSectionOptions* _crossSectionOptions
	);

	virtual void rendering() override;


	template <typename CrossSectionOptionsMode::SpanMode> void traceCrossSectionField();

	void retraceCrossSectionField();

	
};