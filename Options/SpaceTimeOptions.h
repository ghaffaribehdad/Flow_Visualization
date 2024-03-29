#pragma once
#include "RaycastingOptions.h"

namespace SpaceTimeRendering
{
	enum HeightMode
	{
		V_X_FLUCTUATION = 0,
		V_Y_FLUCTUATION,
		V_Z_FLUCTUATION,
		SHEAR_STRESS,
		KINETIC_ENERGY,
		COUNT
	};

	static const char* HeightModeList[]
	{
		"Vx",
		"Vy",
		"Vz",
		"Shear Stress",
		"Kinetic Energy"
	};

}

namespace SliderBackground
{
	enum SliderBackground
	{
		NONE,
		FORWARD,
		BACKWARD,
		BAND,
		COUNT
	};

	static const char* SliderBackgroundList[]
	{
		"None",
		"Forward",
		"Backward",
		"Band"
	};
}



struct SpaceTimeOptions
{

	// Terrain Rendering Options
	int fieldMode = 0;
	float brightness = 1.0f;
	float hegiht_tolerance = 0.01f;
	int wallNoramlPos = 9;
	int timePosition = 2;
	float isoValue = 1;
	float samplingRatio_t =0.01f;
	float heightLimit = 0.5f;
	int spanwisePos = 0;

	// Color Coding
	//int colorCode = static_cast<int>(TimeSpaceRendering::fluctuationColorCode::NONE);
	float max_val = 0.2f;
	float min_val = -.2f;
	float height_scale = 0.0f;
	float offset = 0.0f;

	bool shifSpaceTime = false;
	bool shiftProjection = false;
	float projectionPlanePos = 0.0f;

	float minColor[4] = { 117 / 255.0f,112 / 255.0f,179 / 255.0f };
	float maxColor[4] = { 217 / 255.0f,95 / 255.0f,2 / 255.0f };

	// Rendering status
	bool initialized = false;
	bool dispersion = false;
	bool retrace = false;
	bool released = false;
	bool shading = false;
	bool gaussianFilter = false;
	bool gaussianFilterHeight = false;
	bool resize = false;

	int filterSize = 4;
	float std = 1.0f;


	int filterSizeHeight = 4;
	float stdHeight = 4;

	float	samplingRate_0 = 0.01f;
	bool	usingAbsolute = true;
	int		streamwiseSlice = 1;
	float	streamwiseSlicePos = 0;

	int currentTime = 0;
	int sliderBackground = SliderBackground::SliderBackground::NONE;
	int bandSize = 10;

	int heightMode = IsoMeasure::VELOCITY_X;

	int spaceTimeMode_0 = IsoMeasure::VELOCITY_X;
	int spaceTimeMode_1 = IsoMeasure::VELOCITY_Y;
	int spaceTimeMode_2 = IsoMeasure::VELOCITY_Z;
	int spaceTimeMode_3 = IsoMeasure::KINETIC_ENERGY;


	bool additionalRaycasting = false;
	bool additionalLoading = true;
	bool volumeLoaded = false;


};


