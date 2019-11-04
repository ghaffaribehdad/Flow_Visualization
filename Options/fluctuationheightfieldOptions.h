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


	// Color Coding
	int colorCode = static_cast<int>(fluctuationOptions::fluctuationColorCode::NONE);
	float max_val = 1.0f;
	float min_val = 0.0f;
	float minColor[4] = { 0,1,0,1 };
	float maxColor[4] = { 0,0,1,1 };


	// Rendering status
	bool initialized = false;
	bool released = true;
	bool dispersion = false;
	bool retrace = false;
};


