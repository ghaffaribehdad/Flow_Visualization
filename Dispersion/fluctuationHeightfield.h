#pragma once

#include "DispersionTracer.h"
#include "..//Options/fluctuationheightfieldOptions.h"
#include "..//Raycaster/BoundingBox.h"

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


	void traceFluctuationfield();
	void gradientFluctuationfield();
	void rendering() override;
	__host__ bool initializeBoundingBox() override;

	FluctuationheightfieldOptions* fluctuationOptions;
private:

	bool LoadVelocityfieldPlane(const unsigned int& idx, const int & plane);
	int3 m_gridSize = { 0,0,0 };

};