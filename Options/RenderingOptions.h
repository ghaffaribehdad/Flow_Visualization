#pragma once
//#include "../Graphics/ImGui/imgui.h"
typedef unsigned int uint;

namespace RenderingMode
{
	enum RenderingMode
	{
		TUBES = 0,
		SPHERES,
		COUNT
	};

	static const char* const RenderingModeList[] =
	{
		"Tubes",
		"Spheres",
	};
}
namespace DrawMode
{

	
	enum DrawMode
	{
		STATIONARY = 0,
		ADVECTION,
		CURRENT,
		COUNT
	};

	static const char* const DrawModeList[] =
	{
		"Stationary",
		"Advection",
		"Current"
	};

}


struct RenderingOptions
{
	float tubeRadius = 0.01f;
	float minColor[4] = 
	{ 
		27.0f/255.0f,
		158.0f / 255.0f,
		119.0f / 255.0f,
		1.0f
	};
	float maxColor[4] = 
	{
		217.0f / 255.0f,
		95.0f / 255.0f,
		2.0f / 255.0f,
		1.0f 
	};
	int renderingMode = RenderingMode::RenderingMode::TUBES;
	int drawMode = DrawMode::DrawMode::STATIONARY;

	float maxMeasure = 10.0f;
	float minMeasure = 0.0f;


	bool isRaycasting = false;
	float bgColor[4] = { 1.0f,1.0f,1.0f,0.0f };

	bool showSeedBox = true;
	bool showVolumeBox = true;

	float boxRadius = 0.01f;
	int lineLength = 5;

};