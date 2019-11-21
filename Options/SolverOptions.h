#pragma once
#include "Enum.h"
#include <d3d11.h>

struct SolverOptions
{
public:
	char fileName[100] ="UCLA_CTBL";
	char  filePath[100] ="O:\\Share\\Behdad\\UCLA_Michi\\";


	char filePath_out[100] = "D:\\git_projects\\LineOutput\\";
	char fileName_out[100] = "Lineset";
	int counter = 0;
	int fileToSave = 1000;

	int gridSize[3] = { 130,384,384 };
	int seedGrid[3] = { 5,20,100 };
	float gridDiameter[3] = { 1.7f,5,5 };


	float seedBox[3] = { 1.7f,5,5 };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f };

	int precision = 32;

	float dt = 0.1f;
	float advectTime = 0.0f;

	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 1;
	int currentIdx =1;

	
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