#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>

struct SolverOptions
{
public:
	char fileName[100] ="FieldPadded";
	char  filePath[100] ="I:\\binary_fluct_x_major_Padded_1_1000\\";


	char filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded";
	char fileName_out[100] = "Lineset";
	int counter = 0;
	int fileToSave = 1000;

	int gridSize[3] = { 64,503,2048 };
	int seedGrid[3] = { 5,20,100 };
	float gridDiameter[3] = { 1,2,5 };
	//float gridDiameter[3] = { 7.854f,2.0f,3.1415f };


	//float seedBox[3] = { 7.854f,2.0f,3.1415f };
	float seedBox[3] = { 1,2,5 };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	int precision = 32;

	float dt = 0.001f;
	float advectTime = 0.0f;

	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 750;
	int lastIdx = 950;
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