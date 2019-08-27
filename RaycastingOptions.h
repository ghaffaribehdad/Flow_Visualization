#pragma once
#include "Enum.h"

struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.1f;
	float tolerance_0 = 0.1f;
	float isoValue_0 = 0.0f;

	IsoMeasure isoMeasure_0 = IsoMeasure::VelocityMagnitude;
	IsoMeasure isoMeasure_1 = IsoMeasure::VelocityMagnitude;
	IsoMeasure isoMeasure_2 = IsoMeasure::VelocityMagnitude;


	float color_0[4] = { 0.5f,0.5f,0.5f,1.0f };
};