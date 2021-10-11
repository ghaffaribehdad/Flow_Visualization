#pragma once
#include "../Particle/Particle.h"
#include "../Cuda/CudaArray.h"
#include "../Cuda/cudaSurface.h"
#include "../VolumeTexture/VolumeTexture.h"
#include "../Raycaster/Raycasting.h"

class FSLE : public Raycasting
{

private:

	Particle * h_Particle;
	Particle * d_Particle;
	CudaSurface s_fsle;

public:

	virtual bool initialize(cudaTextureAddressMode, cudaTextureAddressMode, cudaTextureAddressMode);
	__host__ virtual bool release() override;
	__host__ virtual void rendering() override;
	__host__ virtual bool updateScene() override;
	virtual void show(RenderImGuiOptions* renderImGuiOptions) override;
};
