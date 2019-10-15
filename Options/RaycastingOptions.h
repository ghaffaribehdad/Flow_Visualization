#pragma once
#include "Enum.h"

struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.01f;
	float tolerance_0 = 0.001f;
	float isoValue_0 = 10.0f;

	bool fileLoaded = false;
	bool fileChanged = false;
	bool Raycasting = false;
	bool initialized = false;

	int isoMeasure_0 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_1 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_2 = IsoMeasure::VelocityMagnitude;




	float color_0[3] = { 0.5f,0.5f,0.5f};
};