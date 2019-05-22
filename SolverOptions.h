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
	float dt = 0;
	bool begin = false;
	int lines_count = 0;
	int particle_count = 0;
	SeedingPattern seedingPatter;
	IntegrationMethod integrationMethod;
	InterpolationMethod interpolationMethod;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	SolverOptions() {};


};