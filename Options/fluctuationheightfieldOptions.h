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



struct TimeSpaceRenderingOptions
{

	// Terrain Rendering Options
	int fieldMode = 0;

	float hegiht_tolerance = 0.01f;
	int wallNoramlPos = 0;
	int wallNormalgridSize = 250;
	float heightLimit = 0.5f;
	float gridDiameter[3] = {5.0f, 5.0f, 5.0f };
	int spanwisePos = 0;

	// Color Coding
	int colorCode = static_cast<int>(TimeSpaceRendering::fluctuationColorCode::NONE);
	float max_val = 1.0f;
	float min_val = 1.0f;
	float height_scale = 1.0;
	float offset = 0.0f;
	float minColor[4] = { 0,0,1,1 };
	float maxColor[4] = { 1,0,0,1 };


	// Rendering status
	bool initialized = false;
	bool dispersion = false;
	bool retrace = false;
	bool released = false;


	char	fileName[100] = "FieldPadded";
	char	filePath[100] = "Q:\\MinimalChannel\\binary_fluct_x_major_Padded_1_1000\\";
	int		firstIdx = 100;
	int		lastIdx = 120;
	float	samplingRate_0 = 0.01f;
	bool	usingAbsolute = true;
	int		streamwiseSlice = 1;



};


