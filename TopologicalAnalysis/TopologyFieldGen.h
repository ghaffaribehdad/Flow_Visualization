#pragma once

#include "../Cuda/cudaSurface.h"
#include "../VolumeTexture/VolumeTexture.h"

class TopologyFieldGen
{

private:

	CudaSurface cudaSurface;
	VolumeTexture3D_T<float4> volume_texture;

	cudaArray_t a_velocityField;
	cudaArray_t a_topologyField;

public:

	void generateField();
	bool initialization();
	bool release();
};