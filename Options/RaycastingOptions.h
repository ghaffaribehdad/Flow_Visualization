#pragma once
#include "ImGuiOptions.h"

struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.001f;
	float tolerance_0 = 0.001f;
	float isoValue_0 = 10.0f;

	float planeThinkness = 0.002f;

	bool fileLoaded = false;
	bool fileChanged = false;
	bool Raycasting = false;
	bool initialized = false;

	int isoMeasure_0 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_1 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_2 = IsoMeasure::VelocityMagnitude;


	float wallNormalClipping = 0.5f;

	float color_0[3] = { 0.8f,0.8f,0.8f};

	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };

	float minVal = -1.0f;
	float maxVal = +1.0f;

	bool identicalDataset = true;


	char fileName[100] = "timespaceOFstreamwise";
	char filePath[100] = "F:\\Dataset\\KIT3\\binary_fluc_z_major\\OpticalFlowPaddedStreamwise\\";
	int timestep = 1;

};