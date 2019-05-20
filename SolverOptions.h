#pragma once
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
};