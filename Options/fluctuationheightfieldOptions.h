#pragma once

namespace fluctuationOptions
{
	enum fluctuationColorCode
	{
		NONE = 0,
		V_X_FLUCTUATION,
		V_y_FLUCTUATION,
		V_z_FLUCTUATION

	};
}



struct FluctuationheightfieldOptions
{

	// Terrain Rendering Options
	float hegiht_tolerance = 0.01f;
	int wallNoramlPos = 0;
	int wallNormalgridSize = 250;

	int spanwisePos = 0;

	// Color Coding
	int colorCode = static_cast<int>(fluctuationOptions::fluctuationColorCode::NONE);
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
	char	filePath[100] = "E:\\Datasets\\KIT3\\Fields_fld\\binary_fluct_x_major\\";
	int		firstIdx = 200;
	int		lastIdx = 800;
	float	samplingRate_0 = 0.01f;
	bool	usingAbsolute = true;

};


