#pragma once

struct DispersionOptions
{
	float seedWallNormalDist = 0.01f;
	int timeStep = 10;
	int gridSize_2D[2] = { 192,192 };
	float dt = 0.01f;
	bool initialized = false;
	bool dispersion = false;
	bool retrace = false;
};