#pragma once

struct DispersionOptions
{
	float seedWallNormalDist = 0.01f;
	int timeStep = 10;
	int gridSize_2D[2] = { 10,10 };
	float dt = 0.01f;
};