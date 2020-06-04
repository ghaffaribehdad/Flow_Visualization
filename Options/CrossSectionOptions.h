#pragma once

namespace CrossSectionOptionsMode
{
	enum CrossSectionMode
	{
		XY_SECTION = 0,
		XZ_SECTION,
		YZ_SECTION
	};

	enum class SpanMode
	{
		WALL_NORMAL = 0,
		TIME,
		VOL_3D
	};

	enum class VelocityComponent
	{
		V_X = 0,
		V_Y,
		V_Z
	};

}

struct CrossSectionOptions
{
	bool initialized = false;
	bool updateTime = false;
	int slice = 0;

	
	bool filterMinMax = false;

	float min_max_threshold = 0.01f;

	int wallNormalPos = 50;

	float samplingRate = 0.001f;

	CrossSectionOptionsMode::CrossSectionMode crossSectionMode = CrossSectionOptionsMode::CrossSectionMode::XY_SECTION;
	CrossSectionOptionsMode::SpanMode mode = CrossSectionOptionsMode::SpanMode::TIME;

	float max_val = 5.0f;
	float min_val = 5.0f;
	float minColor[4] = { 1.0f,0.0f,0.0f,1.0f };
	float maxColor[4] = { 0.0f,0.0f,1.0f,1.0f };

};