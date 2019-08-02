#pragma once
#include "Enum.h"

struct SolverOptions
{
public:
	char fileName[100] = "FieldP";
	char  filePath[100] = "D:\\test_Flow_VisualizationTool\\";

	int gridSize[3] = { 192,192,192 };
	float gridDiameter[3] = { 10,10,10 };

	int precision = 32;

	float dt = 0.1f;
	float advectTime = 0.0f;

	int lineLength = 1;
	int lines_count = 1;
	float line_thickness = 0.0f;


	int firstIdx = 1;
	int lastIdx = 1;
	int currentIdx = 1;

	SeedingPattern seedingPatter;
	IntegrationMethod integrationMethod;
	InterpolationMethod interpolationMethod;

	int colorMode = 0;

	bool beginStream = false;
	bool beginPath = false;

	bool interOperation = false;

	IDXGIAdapter* p_Adapter;
	ID3D11Resource* p_vertexBuffer;

	bool beginRaycasting = false;
	bool idChange = false;


	bool userInterruption = true;

	bool streamline = true;
	bool pathline = false;


	bool lengthChanged = false;

	SolverOptions() {};


};