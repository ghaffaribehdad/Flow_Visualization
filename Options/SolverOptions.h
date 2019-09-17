#pragma once
#include "Enum.h"
#include <d3d11.h>

struct SolverOptions
{
public:
	char fileName[100] = "FieldInitialPadded";
	char  filePath[100] = "E:\\Datasets\\KIT3\\Fields_fld\\binary_initial\\";

	int gridSize[3] = { 64,503,2048 };
	int seedGrid[3] = { 10,10,10 };
	float gridDiameter[3] = { 1.0f,5.0f,10.0f };


	float seedBox[3] = { 1.0f,5.0f,10.0f };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f };

	int precision = 32;

	float dt = 0.001f;
	float advectTime = 0.0f;

	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 1;
	int currentIdx = 1;

	
	int colorMode = 0;
	int seedingPattern = 0;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool idChange = false;


	SolverOptions() {};

	int projection = Projection::NO_PROJECTION;
	bool periodic = false;


};