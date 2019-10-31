#pragma once

enum dispersionColorCode
{
	NONE = 0,
	V_X_FLUCTUATION,
	V_Y,
	V_Z,
	DEV_Z,
	DEV_ZY,
	DISTANCE,
	DISTANCE_ZY,
	QUADRANT_DEV,

};


struct DispersionOptions
{
	// Seeding Options
	float seedWallNormalDist = .09f;
	int gridSize_2D[2] = { 256,256 }; //TODO:: Add dynamic seed size
	float tilt_deg = 30.0f;


	// Terrain Rendering Options
	float hegiht_tolerance = 0.01f;

	// Binary Search Options
	float binarySearchTolerance = 0.1f;
	int binarySearchMaxIteration = 50;

	// Color Coding
	int colorCode = static_cast<int>(dispersionColorCode::NONE);
	float max_val = 1.0f;
	float min_val = 0.0f;
	float minColor[4] = { 0,1,0,1 };
	float maxColor[4] = { 0,0,1,1 };

	// Advection Options
	int timeStep = 1000;
	int tracingTime =  0;
	float dt = 0.001f;
	int sampling_step = 10;

	// Rendering status
	bool initialized = false;
	bool released = true;
	bool dispersion = false;
	bool retrace = false;
};


