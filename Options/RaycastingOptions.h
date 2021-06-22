#pragma once
#include "ImGuiOptions.h"


namespace IsoMeasure

{
	enum IsoMeasureMode
	{
		VelocityMagnitude = 0,
		Velocity_X,
		Velocity_Y,
		Velocity_Z,
		Velocity_W,
		ShearStress,
		TURBULENT_DIFFUSIVITY,
		LAMBDA2,
		LAMBDA2_VELOCITY_X,
		COUNT
	};

	static const char* const IsoMeasureModes[] =
	{
		"Velocity Magnitude",
		"Velocity X",
		"Velocity Y",
		"Velocity Z",
		"Velocity W",
		"Shear Stress",
		"Turbulent Diffusivity",
		"lambda2",
		"lambda2_velocity",
	};

	enum ProjectionPlane
	{
		XYPLANE,
		YZPLANE,
		ZXPLANE,
		COUNT_PLANE
	};
	static const char* const ProjectionPlaneList[] =
	{
		"XY Plane",
		"ZY Plane",
		"ZX Plane"
	};

}


struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.001f;
	float tolerance_0 = 0.001f;
	float isoValue_0 = 10.0f;
	float isoValue_1 = 10.0f;

	float planeThinkness = 0.002f;

	bool fileLoaded = false;
	bool fileChanged = false;
	bool Raycasting = false;
	bool initialized = false;
	bool planarRaycasting = false;

	int projectionPlane = IsoMeasure::ProjectionPlane::YZPLANE;
	int isoMeasure_0 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_1 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_2 = IsoMeasure::VelocityMagnitude;
	float brightness = 0.8f;

	float clipBox[3] = { 1.0f,1.0f,1.0f };
	float clipBoxCenter[3] = { 0.0f,0.0f,0.0f };

	float wallNormalClipping = 0.5f;

	float color_0[3] = { 117 / 255.0f,112 / 255.0f,179 / 255.0f };
	float color_1[3] = { 217 / 255.0f,95 / 255.0f,2 / 255.0f };

	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };

	float minVal = -1.0f;
	float maxVal = +1.0f;
	float transparecny = 1.0f;
	bool identicalDataset = true;
	bool adaptiveSampling = false;
	bool resize = false;

	char fileName[100] = "timespaceOFstreamwise";
	char filePath[100] = "F:\\Dataset\\KIT3\\binary_fluc_z_major\\OpticalFlowPaddedStreamwise\\";
	int timestep = 1;

};