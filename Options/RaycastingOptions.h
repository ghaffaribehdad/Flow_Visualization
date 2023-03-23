#pragma once
#include "ImGuiOptions.h"


namespace IsoMeasure

{
	enum IsoMeasureMode
	{
		VELOCITY_MAGNITUDE = 0,
		VELOCITY_X,
		VELOCITY_Y,
		VELOCITY_Z,
		VELOCITY_W,
		SHEAR_STRESS,
		KINETIC_ENERGY,
		LAMBDA2,
		DIFFERENCE_X,
		DIFFERENCE_Y,
		DIFFERENCE_Z,
		DIFFERENCE_W,
		AVERAGE_DOUBLE,
		MAXIMUM_DOUBLE,
		MINIMUM_DOUBLE,
		NORMAL_CURVES,
		IMPORTANCE_DIFF_LESS,
		Finite_TIME,
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
		"Kinetic Energy",
		"lambda2",
		"Difference X",
		"Difference Y",
		"Difference Z",
		"Difference W",
		"Average Double",
		"Minimum Double",
		"Maximum Double",
		"Gradient Length",
		"Importance diff less",
		"FTLE",
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

	enum FieldLevel
	{
		L0 = 0,
		L1,
		L2,
		COUNT_LEVEL
	};

	static const char* const FieldLevelList[] =
	{
		"L0",
		"L1",
		"L2"
	};

}

namespace Visitation
{
	enum RaycastingMode
	{
		DVR_VISITATION = 0,
		COUNT
	};

	static const char* const RaycastingModeList[] =
	{
		"DVR visitation"
	};
}

namespace RaycastingMode
{
	enum Mode
	{
		SINGLE = 0,
		DVR,
		DVR_DOUBLE,
		SINGLE_COLORCODED,
		DOUBLE,
		DOUBLE_SEPARATE,
		DOUBLE_ADVANCED,
		DOUBLE_TRANSPARENCY,
		MULTISCALE,
		MULTISCALE_TEMP,
		MULTISCALE_DEFECT,
		PLANAR,
		PLANAR_DOUBLE,
		PROJECTION_FORWARD,
		PROJECTION_BACKWARD,
		PROJECTION_AVERAGE,
		PROJECTION_LENGTH,
		COUNT

	};


	static const char* const modeList[] =
	{
		"Single",
		"DVR",
		"DVR Double",
		"Single Color-coded",
		"Double",
		"Double Separate",
		"Double Advanced",
		"Double Transparency",
		"Multi-scale",
		"Multi-scale-temp",
		"Multi-scale Defect",
		"Planar",
		"Planar Double",
		"Projection Forward",
		"Projection Backward",
		"Projection Average",
		"Projection Length"
	};
}




struct RaycastingOptions
{
public:

	float samplingRate_0 = 0.005f;
	float samplingRate_1 = 0.005f;
	float samplingRate_projection = 0.5f;
	float tolerance_0 = 0.00001f;
	float isoValue_0 = 0.2f;
	float isoValue_1 = 0.2f;
	float isoValue_2 = 10.0f;
		
	float planeThinkness = 0.008f;

	bool Raycasting = false;
	bool initialized = false;
	bool planarRaycasting = false;
	bool multiScaleRaycasting = false;
	bool doubleIso = false;
	bool binarySearch = false;
	bool ColorCodingRange = false;
	bool secondaryRaycasting = false;
	bool adaptiveSampling = false;
	bool insideOnly = false;
	bool resize = false;

	bool diffImportance = false;
	float limit_difference = 0.1f;
	bool secondaryOnly = false;
	bool within = false;

	int maxIteration = 10;
	int projectionPlane = IsoMeasure::ProjectionPlane::YZPLANE;
	int isoMeasure_0 = IsoMeasure::VELOCITY_MAGNITUDE;
	int isoMeasure_1 = IsoMeasure::VELOCITY_MAGNITUDE;
	int isoMeasure_2 = IsoMeasure::VELOCITY_MAGNITUDE;

	int fieldLevel_0 = IsoMeasure::FieldLevel::L0;
	int fieldLevel_1 = IsoMeasure::FieldLevel::L1;
	int fieldLevel_2 = IsoMeasure::FieldLevel::L2;

	int raycastingField_0 = 0;
	int raycastingField_1 = 1;

	float reflectionCoefficient = 0.8f;

	float brightness = 0.8f;

	float clipBox[3] = { 1.0f,1.0f,1.0f };
	float clipBoxCenter[3] = { 0.0f,0.0f,0.0f };

	float planeProbePosition = 0.5f;
	float projectionPlanePos = 32.0f;

	float color_0[3] = { 117 / 255.0f,112 / 255.0f,179 / 255.0f };
	float color_1[3] = { 217 / 255.0f,95 / 255.0f,2 / 255.0f };

	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };

	float minVal = 0.0f;
	float maxVal = +1.0f;

	float transparency_0 = 0.0f;
	float transparency_1 = 0.3f;

	float amplitude = 1.0f;
	float relax = 1.0f;

	bool normalCurves = false;
	float max_distance = 0;

	int timestep = 1;
	int raycastingMode = RaycastingMode::Mode::SINGLE;

};