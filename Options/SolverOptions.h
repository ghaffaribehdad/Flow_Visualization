#pragma once
#include "ImGuiOptions.h"
#include <d3d11.h>
#include <string>



namespace ComputationMode
{
	enum ComputationMode
	{
		ONTHEFLY,
		PRECOMPUTATION,
		COUNT

	};

	static const char* ComputationModeList[]
	{
		"On the fly",
		"Pre Computation"
	};
}

namespace TransparencyMode
{
	enum TransparencyMode
	{
		STREAKPOS = 0,
		TIMEPOS,
		COUNT
	};

	static const char* TransparencyModeList[]
	{
		"Streak Pos.",
		"Time Pos",
	};
}

struct SolverOptions
{
public:


	std::string  fileName ="FieldP";
	std::string  filePath = "G:\\KIT3MipMapL1\\";

	bool usingOIT = true;

	float seedWallNormalDist = 0.1f;
	float tilt_deg = 15.0f;

	std::string  outputFileName= "screen";

	char filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded\\";
	char fileName_out[100] = "Lineset";
	int counter;
	int fileCounter = 0;
	int fileToSave = 1000;
	int seedValue = 1;
	bool randomSeed = true;
	int channels = 4;
	int gridSize[3] = { 64,503,2048 };						//KIT3
	//int gridSize[3] = { 100,503,500 };					//KIT3 timespace short spanwise
	float velocityScalingFactor[3] = { 1.0f,1.0f,1.0f};		// time-space scale
	//int gridSize[3] = { 32,1024,1024 };					//TUI ra1e5

	bool viewChanged = true;

	int seedGrid[3] = { 5,5,10 };
	int gridSize_2D[2] = { 192,192 };

	float gridDiameter[3] = { 0.4f,2.0f,7.0f };							//KIT3
	//float gridDiameter[3] = { 2.0f,2.0f,0.5f };						//Schumacher
	//float gridDiameter[3] = { 0.4f,4.0f,4.0f };						//TUI ra1e5
	
	float volumeDiameter[3] = { 0.4f,2.0f,7.0f };

	float seedBox[3] = { 0.4f,2.0f,7.0f };					//KIT3
	float streakBox[3] = { 0.0f,2.0f,7.0f };					//KIT3
	float streakBoxPos[3] = { 0.0f, 0.0f, 0.0f };
	float seedBoxPos[3] = { 0.0f, 0.0f, 0.0f }; 

	bool orderIndependetTransparency = true;

	int precision = 32;

	float dt = 0.001f;
	float advectTime = 0.0f;

	int computationMode = ComputationMode::ComputationMode::PRECOMPUTATION;

	int lineLength = 1024;
	int lines_count = 1024;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 600;
	int currentIdx =1;
	int timeSteps = lastIdx - firstIdx + 1;
	int currentSegment = 0;

	int lineRenderingMode = LineRenderingMode::LineRenderingMode::STREAMLINES;
	int colorMode = 0;
	
	int transparencyMode = TransparencyMode::TransparencyMode::STREAKPOS;
	SeedingPattern seedingPattern = SeedingPattern::SEED_RANDOM;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool idChange = false;
	bool fileChanged = false;
	bool loadNewfile = false;
	bool updatePause = false;
	bool fileLoaded = false;
	bool shutdown = false;
	bool usingTransparency = false;
	bool usingThreshold = false;
	float transparencyThreshold = 0.4f;

	SolverOptions() {};

	int projection = Projection::Projection::NO_PROJECTION;
	bool projectToInit = true;
	int  projectPos = 32;

	bool syncWithStreak = false;

	bool periodic = false;

	bool Compressed = false;

	bool drawComplete = false;

	size_t maxSize = 64000000;

	float timeDim = 5;
};

struct FieldOptions
{
	char	filePath_out[100] = "D:\\copy\\bf\\Binary_z_Major\\Padded\\";
	float	gridDiameter[3] = { 0.4f,2.0f,7.0f };								//KIT3
	int		gridSize[3] = { 64,503,2048 };										//KIT3
	float	dt = 0.001f;
	int		firstIdx = 1;
	int		lastIdx = 600;
	int		currentIdx = 1;
	size_t	maxSize = 64000000;
	bool	periodic = false;
	bool	Compressed = false;
};