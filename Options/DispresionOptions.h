#pragma once
#include <string>

namespace dispersionOptionsMode
{
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

	enum HeightfieldRenderingMode
	{
		SINGLE_SURFACE = 0,
		DOUBLE_SURFACE
	};
}



struct DispersionOptions
{
	// Seeding Options
	int gridSize_2D[2] = { 192,192 }; //TODO:: Add dynamic seed size
	float seedWallNormalDist = 0.20f;
	//float seedWallNormalDist = 0.20f;
	//float seedWallNormalDist = 0.30f;
	//float seedWallNormalDist = 0.50f;

	float tilt_deg = 0.0f;


	char	fileNameSecondary[100] = "FieldP";
	char	filePathSecondary[100] = "G:\\OscillatingWall\\Padded\\";

	float transparencySecondary = 0.5f;

	// Terrain Rendering Options
	float hegiht_tolerance = 0.01f;

	// Distance of the neighboring particle in FTLE calculation
	float initial_distance = 0.0001f;

	// Binary Search Options
	float binarySearchTolerance = 0.1f;
	int binarySearchMaxIteration = 50;

	// Color Coding
	int colorCode = static_cast<int>(dispersionOptionsMode::dispersionColorCode::NONE);
	int renderingMode = static_cast<int>(dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE);
	float max_val = 4.0f;
	float min_val = 0.0f; 
	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };


	float minColor_secondary[4] = { 67.0f / 255.0f,162.0f/ 255.0f,202.0f / 255.0f };
	float maxColor_secondary[4] = { 168.0f / 255.0f,221.0f / 255.0f,181.0f / 255.0f };

	// Advection Options
	int timestep = 1;
	float dt = 0.001f;
	int sampling_step = 10;

	bool marching = false;

	// Rendering status
	bool initialized = false;
	bool released = false;
	bool dispersion = false;
	bool retrace = false;


	// To save screenshots:
	
	std::string fileName = "test";
	std::string filePath = "G:\\Screenshots\\";
	int file_counter = 0;

	bool saveScreenshot = false;
	bool ftleIsosurface = false;

	float ftleIsoValue = 0.0f;

	float isoValueTolerance = 0.01f;
	bool forward = true;
	bool backward = false;

};


