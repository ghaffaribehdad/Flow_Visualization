#pragma once

#include "DispersionTracer.h"

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


};