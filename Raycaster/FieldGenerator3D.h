#pragma once

#include <string>
#include "..//VolumeIO/Volume_IO.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"
#include "..//Raycaster/Raycasting.h"
#include "..//Options/TimeSpace3DOptions.h"
#include "..//Options/SolverOptions.h"


class FieldGenerator3D : public Raycasting
{
public:

virtual void show(RenderImGuiOptions* renderImGuiOptions) = 0;


protected:

	bool traced = false;

	int3 generatedFieldSize = { 0,0,0 };

	virtual bool initialize(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);


	virtual bool cudaRaycaster() = 0;


	bool retrace();
	bool release() override;

	virtual bool updateScene() override;

	void rendering() override;

	// Cuda Surface and Texture to be bound to the Cuda Array of time-space 3D field
	CudaSurface cudaSurface;
	// Cuda Array to store time-space 3D field
	CudaArray_3D<float> cudaArray3D;
	VolumeTexture3D_T<float> cudaTexture3D_float;



	virtual bool generateVolumetricField
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	) = 0;
	virtual bool regenerateVolumetricField() = 0;

	virtual bool InitializeSurface3D();

};



