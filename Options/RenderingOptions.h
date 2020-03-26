#pragma once
//#include "../Graphics/ImGui/imgui.h"
typedef unsigned int uint;


enum RenderMode
{
	LINES = 0,
	TUBES = 1,
};

struct RenderingOptions
{
	float tubeRadius = 0.01f;
	float minColor[4] = { 1.0f,0.0f,0.0f,1.0f};
	float maxColor[4] = { 0.0f,1.0f,0.0f,1.0f};
	RenderMode renderMode = RenderMode::TUBES;

	float maxMeasure = 10.0f;
	float minMeasure = 0.0f;

	bool isRaycasting = false;
	float bgColor[4] = { 1.0f,1.0f,1.0f,1.0f };

	bool showSeedBox = true;
	bool showVolumeBox = true;

};