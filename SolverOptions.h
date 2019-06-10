#pragma once
#include "Enum.h"

struct SolverOptions
{
public:
	char fileName[100] = "";
	char  filePath[100] = "";

	int gridSize[3] = { 192,192,192 };
	float gridDiameter[3] = { 10,10,10 };

	int precision = 32;

	int timestep = 1;
	float dt = 0.1f;
	float advectTime = 0.0f;

	int lineLength;
	int lines_count = 0;
	float line_thickness = 0.0f;

	int firstIdx;
	int lastIdx;
	int currentIdx;

	int particle_count = 10;

	SeedingPattern seedingPatter;
	IntegrationMethod integrationMethod;
	InterpolationMethod interpolationMethod;


	bool begin = false;


	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;




	SolverOptions() {};


};