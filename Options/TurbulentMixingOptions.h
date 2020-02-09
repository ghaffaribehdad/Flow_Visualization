#pragma once

struct TurbulentMixingOptions
{

	float linearCreationRatio = 1.0f;
	float LinearDissipationRatio = 1.0f;

	int streamwisePlane = 0;

	int minVal = -5;
	int maxVal = 5;

	float minColor[4] = { 1.0f,0.0f,0.0f,1.0f };
	float maxColor[4] = { 0.0f,0.0f,1.0f,1.0f };


	bool initialized = false;
};