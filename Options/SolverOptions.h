#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>

struct SolverOptions
{
public:
	char fileName[100] ="MipmapbinaryFloatFieldL3";
	char  filePath[100] ="F:\\LargeRayleigh-Benard\\MipMap_L3\\";


	char filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded\\";
	char fileName_out[100] = "Lineset";
	int counter = 0;
	int fileToSave = 1000;

	int gridSize[3] = { 1200,1200,80 };
	int seedGrid[3] = { 2,2,1 };
	//float gridDiameter[3] = { 1,2,5 };
	//float gridDiameter[3] = { 7.854f,2.0f,3.1415f };	//KIT2
	float gridDiameter[3] = { 2.0f,2.0f,1.0f };	//Schumacher


	float seedBox[3] = { 2.0f,2.0f,2.0f };			//KIT2
	//float seedBox[3] = { 1,2,5 };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	int precision = 32;

	float dt = 0.001f;
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