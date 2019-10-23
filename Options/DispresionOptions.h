#pragma once

struct DispersionOptions
{
	float seedWallNormalDist = .09f;
	int gridSize_2D[2] = { 192,192 };
	float dt = 0.001f;
	bool initialized = false;
	bool dispersion = false;
	bool retrace = false;
	float hegiht_tolerance = 0.05f;
	float dev_z_range = 1.0f;
	float dev_mag_levels = 5.0f;
	float dev_mag_tolerance = 0.1f;
	float binarySearchTolerance = 0.1f;
	int binarySearchMaxIteration = 50;


	int timeStep = 1000;
	int tracingTime =  1024;
};