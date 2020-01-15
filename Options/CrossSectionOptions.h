#pragma once

namespace CrossSectionOptionsMode
{
	enum CrossSectionMode
	{
		XY_SECTION = 0,
		XZ_SECTION,
		YZ_SECTION
	};

}

struct CrossSectionOptions
{
	bool initialized = false;
	bool updateCrossSection = false;
	int slice = 0;
	float samplingRate = 0.001f;

	int crossSectionMode = static_cast<int>(CrossSectionOptionsMode::CrossSectionMode::XY_SECTION);

	float max_val = 1.0f;
	float min_val = 0.0f;
	float minColor[4] = { 0.0f,1.0f,0.0f,1.0f };
	float maxColor[4] = { 0.0f,0.0f,1.0f,1.0f };

};