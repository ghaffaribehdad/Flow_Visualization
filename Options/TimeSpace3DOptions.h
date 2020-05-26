#pragma once

struct TimeSpace3DOptions
{
public:

	int t_first = 1;
	int t_last = 400;
	int streamwisePos = 10;
	float boundingBoxDimensions[3] = { 7.854f,2.0f,3.1415f };
	float wallNormalClipping = 0.5f;
	bool initialized = false;

	float isoValueTolerance = 0.1f;

	float isoValue = 100.0f;
	float minVal = 50.0f;
	float maxVal = 50.0f;

	float samplingRate = 0.005f;
	float tolerance = 0.002f;
	int iteration = 50;
	float color[3] = { 0.8f,0.8f,0.8f };

	float minColor[4] = { 5 / 255.0f,113 / 255.0f,176 / 255.0f };
	float maxColor[4] = { 202 / 255.0f,0 / 255.0f,32 / 255.0f };


};