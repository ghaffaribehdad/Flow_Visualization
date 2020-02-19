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
	float seedWallNormalDist = 0.01f;
	int gridSize_2D[2] = { 256,256 }; //TODO:: Add dynamic seed size
	float tilt_deg = 10.0f;


	char	fileNameSecondary[100] = "FieldP";
	char	filePathSecondary[100] = "G:\\OscillatingWall\\Padded\\";

	float transparencySecondary = 0.5f;

	// Terrain Rendering Options
	float hegiht_tolerance = 0.01f;

	// Distance of the neighboring particle in FTLE calculation
	float ftleDistance = 0.001f;

	// Binary Search Options
	float binarySearchTolerance = 0.1f;
	int binarySearchMaxIteration = 50;

	// Color Coding
	int colorCode = static_cast<int>(dispersionOptionsMode::dispersionColorCode::NONE);
	int renderingMode = static_cast<int>(dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE);
	float max_val = 1.0f;
	float min_val = 0.0f;
	float minColor[4] = { 0.0f,1.0f,0.0f,1.0f };
	float maxColor[4] = { 0.0f,0.0f,1.0f,1.0f };


	float minColor_secondary[4] = { 1.0f,0.0f,0.0f,1.0f };
	float maxColor_secondary[4] = { 1.0f,0.0f,0.0f,1.0f };

	// Advection Options
	int timestep = 1;
	float dt = 0.001f;
	int sampling_step = 10;

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

};


