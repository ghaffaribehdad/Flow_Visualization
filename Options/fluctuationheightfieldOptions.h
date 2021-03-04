#pragma once
#include "RaycastingOptions.h"

namespace TimeSpaceRendering
{


	enum fluctuationColorCode
	{
		NONE = 0,
		V_X_FLUCTUATION,
		V_y_FLUCTUATION,
		V_z_FLUCTUATION

	};

	enum FieldMode
	{
		FF_VX_VY,
		FF_VZ_VY,
		FI_VX_VY
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
	int colorCode = static_cast<int>(TimeSpaceRendering::fluctuationColorCode::NONE);
	float max_val = 1.0f;
	float min_val = 1.0f;
	float height_scale = 0.0f;
	float offset = 0.0f;
	float minColor[4] = { 0,0,1,1 };
	float maxColor[4] = { 1,0,0,1 };


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



};


