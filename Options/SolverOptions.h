#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>
#include <string>

struct SolverOptions
{
public:


	std::string fileName ="FieldP";
	
	// KIT3 Optical Flow on Streamwise timespace field
	std::string  filePath= "G:\\KIT3MipMapL1\\";



	float seedWallNormalDist = 0.1f;
	float tilt_deg = 15.0f;

	char filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded\\";
	char fileName_out[100] = "Lineset";
	int counter = 0;
	int fileToSave = 1000;
	
	int channels = 4;
	int gridSize[3] = { 64,503,2048 };						//KIT3
	//int gridSize[3] = { 100,503,500 };					//KIT3 timespace short spanwise
	float velocityScalingFactor[3] = { 1.0f,1.0f,1.0f};		// time-space scale
	//int gridSize[3] = { 32,1024,1024 };					//TUI ra1e5



	int seedGrid[3] = { 5,5,10 };
	int gridSize_2D[2] = { 192,192 };

	float gridDiameter[3] = { 0.4f,2.0f,7.0f };							//KIT3
	//float gridDiameter[3] = { 2.0f,2.0f,0.5f };						//Schumacher
	//float gridDiameter[3] = { 0.4f,4.0f,4.0f };						//TUI ra1e5
	
	float volumeDiameter[3] = { 0.4f,2.0f,7.0f };

	float seedBox[3] = { 0.4f,2.0f,7.0f };					//KIT3


	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	int precision = 32;

	float dt = 0.001f;
	float advectTime = 0.0f;



	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 600;
	int currentIdx =1;
	int timeSteps = lastIdx - firstIdx + 1;

	int lineRenderingMode = 0;
	int colorMode = 0;
	SeedingPattern seedingPattern = SeedingPattern::SEED_RANDOM;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool idChange = false;
	bool fileChanged = false;
	bool fileLoaded = false;
	bool shutdown = false;

	SolverOptions() {};

	int projection = Projection::NO_PROJECTION;
	bool periodic = false;




};