#pragma once

typedef unsigned int uint;


enum RenderMode
{
	LINES = 0,
	TUBES = 1,
};

struct RenderingOptions
{
	float tubeRadius = 1.0f;
	RenderMode renderMode = RenderMode::TUBES;
};