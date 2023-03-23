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
		FULL = 0,
		CURRENT,
		CURRENT_FULL,
		ADVECTION,
		ADVECTION_FINAL,
		COUNT
	};

	static const char* DrawModeList[] =
	{
		"Full",
		"Current",
		"Current Full",
		"Advection",
		"Adv. Final"
	};

}


struct RenderingOptions
{
	float mouseSpeed = 0.005f;
	float tubeRadius = 0.005f;
	float nearField = 0.001f;
	float farField = 100;
	float minColor[4] = 
	{ 
		178.0f/255.0f,
		136.0f / 255.0f,
		103.0f/ 255.0f,
		1.0f
	};	
	float FOV_deg = 30.0f;

	float maxColor[4] =
	{
	27.0f / 255.0f,
	158.0f / 255.0f,
	119.0f / 255.0f,
	1.0f
	};

	int renderingMode = RenderingMode::RenderingMode::TUBES;
	int drawMode = DrawMode::DrawMode::FULL;

	float maxMeasure = 10.0f;
	float minMeasure = 0.0f;


	float bgColor[4] = {1.0f,1.0f,1.0f,1.0f };
	float lightColor[4] = {1.0f,1.0f,1.0f,1.0f };

	bool showSeedBox = true;
	bool showStreakBox = false;
	bool showStreakPlane = false;
	bool showVolumeBox = true;
	bool showClipBox = true;

	float boxRadius = 0.005f;
	int lineLength = 1;

	float Ka	= 0.0f;
	float Kd	= 1.0f;
	float Kd1	= 1.0f;
	float Ks	= 0.0f;
	float Ks1	= 0.0f;
	float shininess = 2;

};