#pragma once
#include "Enum.h"

struct SolverOptions
{
	char fileName[100] = "";
	char  filePath[100] = "";

	int gridSize[3] = { 0,0,0 };
	float gridDiameter[3] = { 0,0,0 };

	int precision = 0;

	int timestep = 0;
	float dt = 0.0f;
	float advectTime = 0.0f;

	int lineLength;
	int lines_count = 0;
	float line_thickness = 0.0f;

	int particle_count = 0;

	SeedingPattern seedingPatter;
	IntegrationMethod integrationMethod;
	InterpolationMethod interpolationMethod;


	bool begin = false;


	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;




	SolverOptions() {};


};