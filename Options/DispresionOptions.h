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
	float hegiht_tolerance = 0.05f;
	float dev_z_range = 10.0f;
	float dev_mag_levels = 5.0f;
	float dev_mag_telorance = 0.1f;
};