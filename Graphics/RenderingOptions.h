#pragma once
#include "ImGui/imgui.h"
typedef unsigned int uint;


enum RenderMode
{
	LINES = 0,
	TUBES = 1,
};

struct RenderingOptions
{
	float tubeRadius = 0.01f;
	float minColor[4] = { 1,0,0,1 };
	float maxColor[4] = { 0,1,0,1};
	RenderMode renderMode = RenderMode::TUBES;

	float maxMeasure = 10;
	float minMeasure = 0;

	bool isRaycasting = false;
};