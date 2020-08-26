#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>
#include <string>

struct SolverOptions
{
public:

	//char	fileName[100] = "FieldPadded";													// X_Major KIT3
	//char	filePath[100] = "Q:\\MinimalChannel\\binary_fluct_x_major_Padded_1_1000\\";		// X_Major KIT3
	//char fileName[100] ="FieldP";															// Optical Flow Motion Field
	//char fileName[100] ="OF_temperature";													// TUI OF ra1e5	 
	//char fileName[100] ="avg_OF_temperature_300_wsize100_";									// TUI  ra1e5 OF wsize_100	 
	//char fileName[100] ="Field";															// TUI ra1e5	 
	//char fileName[100] ="of_streamwise";
	std::string fileName ="FieldP";
	//char  filePath[100] ="D:\\copy\\ref\\Binary_z_Major\\Padded\\";						// Reference Dataset
	//char  filePath[100] ="G:\\OscillatingWall\\Padded\\";									// Oscillating wall
	//char  filePath[100] ="D:\\copy\\bf\\Binary_z_Major\\Padded\\";						// Virtual body		
	//char  filePath[100] ="G:\\OF_KIT3\\";													// Optical Flow Motion field	
	//char  filePath[100] ="G:\\KIT3_ZMajor_Padded\\";										// KIT3		
	//char  filePath[100] ="E:\\TUI\\of_tui_ra1e5\\";										// TUI OF ra1e5	
	//char  filePath[100] ="E:\\TUI\\tui_ra1e5\\";											// TUI  ra1e5	
	//char  filePath[100] ="E:\\TUI\\of_tui_ra1e5\\avg_of_temp_wsize_100\\";				// TUI  ra1e5 OF wsize_100	
	//char  filePath[100] ="F:\\Dataset\\KIT3\\binary_fluc_z_major\\";						// KIT3 Optical Flow on Streamwise 
	
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
	//int gridSize[3] = { 192,192,192 };					//KIT2
	//int gridSize[3] = { 32,1024,1024 };					//TUI ra1e5



	int seedGrid[3] = { 5,5,10 };
	int gridSize_2D[2] = { 192,192 };

	float gridDiameter[3] = { 0.4f,2.0f,7.0f };							//KIT3
	//float gridDiameter[3] = { 5.0f,2.0f,7.0f };						//Timespace OF field
	//float gridDiameter[3] = { 7.854f,2.0f,3.1415f };					//KIT2
	//float gridDiameter[3] = { 2.0f,2.0f,0.5f };						//Schumacher
	//float gridDiameter[3] = { 0.4f,4.0f,4.0f };						//TUI ra1e5
	
	float volumeDiameter[3] = { 0.4f,2.0f,7.0f };

	//float seedBox[3] = { 7.854f,2.0f,3.1415f };			//KIT2
	float seedBox[3] = { 0.4f,2.0f,7.0f };					//KIT3


	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	int precision = 32;

	//float dt = 0.001f;
	float dt = 0.5f; //	TUI ra1e5
	float advectTime = 0.0f;



	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 600;
	int currentIdx =1;
	int timeSteps = lastIdx - firstIdx + 1;

	
	int colorMode = 0;
	SeedingPattern seedingPattern = SeedingPattern::SEED_RANDOM;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool idChange = false;
	bool fileChanged = false;
	bool fileLoaded = false;

	bool opticalFlow = false;

	bool shutdown = false;

	SolverOptions() {};

	int projection = Projection::NO_PROJECTION;
	bool periodic = false;




};