#pragma once

struct PathSpaceTimeOptions{

	int seedGrid[3] = { 1024,1024,100 };
	int timeGrid = 20;
	int firstIdx = 1;
	int lastIdx = 601;

	int timeStep = 0;
	bool initialized = false;
	bool colorCoding = false;
	int minimumDim = 0;

	bool dispersion = false;

	float sigma = 1.0f;

	int isoMeasure_0 = IsoMeasure::Finite_TIME;
};