#pragma once

typedef unsigned int uint;


enum RenderMode
{
	LINES = 0,
	TUBES = 1,
};

struct RenderingOptions
{
	float tubeRadius = 0.1f;
	RenderMode renderMode = RenderMode::TUBES;
};