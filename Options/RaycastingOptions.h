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
		LAMBDA2,
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
		"lambda2"
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


namespace RaycastingMode
{
	enum Mode
	{
		SINGLE = 0,
		DOUBLE,
		DOUBLE_SEPARATE,
		MULTISCALE,
		PLANAR,
		COUNT

	};

	static const char* const modeList[] =
	{
		"Single",
		"Double",
		"Double Separate",
		"Multi-scale",
		"Planar"
	};
}



struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.0001f;
	float samplingRate_1 = 0.001f;
	float tolerance_0 = 0.0001f;
	float isoValue_0 = 10.0f;
	float isoValue_1 = 10.0f;
	float isoValue_2 = 10.0f;

	float planeThinkness = 0.008f;

	bool fileLoaded = false;
	bool fileChanged = false;
	bool Raycasting = false;
	bool initialized = false;
	bool planarRaycasting = false;
	bool multiScaleRaycasting = false;
	bool doubleIso = false;
	bool binarySearch = false;

	int maxIteration = 10;
	int projectionPlane = IsoMeasure::ProjectionPlane::YZPLANE;
	int isoMeasure_0 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_1 = IsoMeasure::VelocityMagnitude;
	int isoMeasure_2 = IsoMeasure::VelocityMagnitude;
	//int isoMeasure_2 = IsoMeasure::VelocityMagnitude;
	float brightness = 0.8f;

	float clipBox[3] = { 1.0f,1.0f,1.0f };
	float clipBoxCenter[3] = { 0.0f,0.0f,0.0f };

	float planeProbePosition = 0.5f;

	float color_0[3] = { 117 / 255.0f,112 / 255.0f,179 / 255.0f };
	float color_1[3] = { 217 / 255.0f,95 / 255.0f,2 / 255.0f };

	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };

	float minVal = 0.0f;
	float maxVal = +1.0f;
	float transparecny = 0.5f;
	bool adaptiveSampling = false;
	bool resize = false;

	char fileName[100] = "timespaceOFstreamwise";
	char filePath[100] = "F:\\Dataset\\KIT3\\binary_fluc_z_major\\OpticalFlowPaddedStreamwise\\";
	int timestep = 1;
	int raycastingMode = RaycastingMode::Mode::PLANAR;

};