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

	static const char* RenderingModeList[] =
	{
		"Tubes",
		"Spheres",
	};
}



namespace DrawMode
{

	
	enum DrawMode
	{
		REALTIME = 0,
		STATIONARY,
		CURRENT,
		ADVECTION,
		ADVECTION_FINAL,
		COUNT
	};

	static const char* DrawModeList[] =
	{
		"Realtime",
		"Strationary",
		"Current",
		"Advection",
		"Adv. Final"
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
	27.0f / 255.0f,
	158.0f / 255.0f,
	119.0f / 255.0f,
	1.0f
	};
	//{
	//	217.0f / 255.0f,
	//	95.0f / 255.0f,
	//	2.0f / 255.0f,
	//	1.0f 
	//};
	int renderingMode = RenderingMode::RenderingMode::TUBES;
	int drawMode = DrawMode::DrawMode::REALTIME;

	float maxMeasure = 10.0f;
	float minMeasure = 0.0f;


	bool isRaycasting = false;
	float bgColor[4] = { 1.0f,1.0f,1.0f,1.0f };

	bool showSeedBox = true;
	bool showVolumeBox = true;
	bool showClipBox = true;

	float boxRadius = 0.005f;
	int lineLength = 1;

};