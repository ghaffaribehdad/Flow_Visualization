#pragma once

#include "cuda_runtime.h"
#include "..//Enum.h"
#include "volumeTexture.h"


class CUDARaycaster
{

private:
	cudaSurfaceObject_t TMP = NULL;		// front face
	cudaSurfaceObject_t DIR = NULL;		// back face

	float3 eyePos = { 0,0,0 };
	float3 viewDir = { 0,1,0};
	float3 upVector = { 0,0,1 };

	float foVY = 1.0;
	float foV = 90.0;
	int2 resoultion = {1024,768};
	float aspectRatio = 1;


	Isosurface mode = Isosurface::VELOCITY_X;
	int samplingRate;
	int2 resolution;

public:
	bool InitializeSurface();

	//setter and getter
	void setMode(Isosurface _mode);
	void setResolution(const int2& _resoultion);
	void setSamplingRate(const int& _samplingRate);




};