#pragma once
#include "RaycastingOptions.h"

namespace TimeSpaceRendering
{
	enum HeightMode
	{
		V_X_FLUCTUATION = 0,
		V_Y_FLUCTUATION,
		V_Z_FLUCTUATION,
		COUNT
	};

	static const char* HeightModeList[]
	{
		"Vx",
		"Vy",
		"Vz",
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



struct TimeSpaceRenderingOptions
{

	// Terrain Rendering Options
	int fieldMode = 0;

	float hegiht_tolerance = 0.01f;
	int wallNoramlPos = 9;
	float heightLimit = 0.5f;
	int spanwisePos = 0;

	// Color Coding
	//int colorCode = static_cast<int>(TimeSpaceRendering::fluctuationColorCode::NONE);
	float max_val = 1.0f;
	float min_val = 1.0f;
	float height_scale = 0.0f;
	float offset = 0.0f;


	float minColor[4] = { 117 / 255.0f,112 / 255.0f,179 / 255.0f };
	float maxColor[4] = { 217 / 255.0f,95 / 255.0f,2 / 255.0f };


	// Rendering status
	bool initialized = false;
	bool dispersion = false;
	bool retrace = false;
	bool released = false;
	bool shading = false;
	bool gaussianFilter = false;


	float	samplingRate_0 = 0.01f;
	bool	usingAbsolute = true;
	int		streamwiseSlice = 1;
	float	streamwiseSlicePos = 0;

	int currentTime = 0;
	int sliderBackground = SliderBackground::SliderBackground::NONE;
	int bandSize = 10;

	int heightMode = TimeSpaceRendering::HeightMode::V_Y_FLUCTUATION;

	bool additionalRaycasting = false;
	bool additionalLoading = true;


};


