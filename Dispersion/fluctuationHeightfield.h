#pragma once

#include "DispersionTracer.h"
#include "..//Options/fluctuationheightfieldOptions.h"
#include "..//Raycaster/BoundingBox.h"

struct size_t3
{
	size_t x;
	size_t y;
	size_t z;
};

class FluctuationHeightfield : public HeightfieldGenerator
{
	// 
public:
	bool initialize
	(
		cudaTextureAddressMode addressMode_X,
		cudaTextureAddressMode addressMode_Y,
		cudaTextureAddressMode addressMode_Z
	) override;



	void traceFluctuationfield3D();
	void gradientFluctuationfield();
	virtual void rendering() override;
	virtual bool InitializeHeightTexture3D_Single() override;
	__host__ bool initializeBoundingBox() override;


	FluctuationheightfieldOptions* fluctuationOptions;
private:
	size_t3 m_gridSize3D = { 0,0,0 };
	int2 m_gridSize2D = { 0,0 };
	CudaArray_2D<float> heightArray2D;
	cudaTextureObject_t heightFieldTexture2D;

	virtual bool InitializeHeightSurface3D_Single() override;
	virtual bool InitializeHeightArray3D_Single(int3 gridSize) override;


};





