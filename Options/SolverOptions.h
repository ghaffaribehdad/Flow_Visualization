#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>

struct SolverOptions
{
public:
	char fileName[100] ="FieldP";
	//char  filePath[100] ="D:\\copy\\ref\\Binary_z_Major\\Padded\\";	// Reference Dataset
	char  filePath[100] ="G:\\OscillatingWall\\Padded\\";				// Oscillating wall
	//char  filePath[100] ="D:\\copy\\bf\\Binary_z_Major\\Padded\\";			// Virtual body		


	float seedWallNormalDist = 0.1f;
	float tilt_deg = 15.0f;

	char filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded\\";
	char fileName_out[100] = "Lineset";
	int counter = 0;
	int fileToSave = 1000;

	int channels = 4;
	int gridSize[3] = { 192,192,192 };
	int seedGrid[3] = { 2,2,1 };
	int gridSize_2D[2] = { 50,50 };

	//float gridDiameter[3] = { 1,2,5 };
	float gridDiameter[3] = { 7.854f,2.0f,3.1415f };	//KIT2
	//float gridDiameter[3] = { 2.0f,2.0f,0.5f };	//Schumacher
	

	float seedBox[3] = { 7.854f,2.0f,3.1415f };			//KIT2
	//float seedBox[3] = { 1,2,5 };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	int precision = 32;

	float dt = 0.001f;
	float advectTime = 0.0f;



	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 400;
	int lastIdx = 750;
	int currentIdx =400;
	int timeSteps = lastIdx - firstIdx + 1;

	
	int colorMode = 0;
	SeedingPattern seedingPattern = SeedingPattern::SEED_RANDOM;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool idChange = false;


	SolverOptions() {};

	int projection = Projection::NO_PROJECTION;
	bool periodic = false;




};