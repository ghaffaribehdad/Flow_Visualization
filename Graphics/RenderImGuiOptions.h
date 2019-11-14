#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"
#include "RenderingOptions.h"
#include "Camera.h"
#include "../Timer/Timer.h"
#include <map>
#include <string>


class RenderImGuiOptions
{



private:

	Camera				*	camera;
	Timer				*	fpsTimer;
	SolverOptions		*	solverOptions;
	RenderingOptions	*	renderingOptions;
	RaycastingOptions	*	raycastingOptions;
	DispersionOptions	*	dispersionOptions;
	FluctuationheightfieldOptions* fluctuationOptions;
	


public:

	void setResources
	(
		Camera* _camera,
		Timer * _fpsTimer, 
		RenderingOptions * _renderingOptions,
		SolverOptions * _solverOptions,
		RaycastingOptions * _raycastingOptions,
		DispersionOptions * _dispersionOptions,
		FluctuationheightfieldOptions * _fluctuationheightfieldOptions
	)
	{
		this->camera = _camera;
	
		this->fpsTimer = _fpsTimer;
	
		this->renderingOptions = _renderingOptions;
	
		this->solverOptions = _solverOptions;

		this->raycastingOptions = _raycastingOptions;

		this->dispersionOptions = _dispersionOptions;

		this->fluctuationOptions = _fluctuationheightfieldOptions;
	}

	bool updateRaycasting		= false;
	bool updateLineRendering	= false;
	bool updateSeedBox			= false;
	bool updateVolumeBox		= false;
	bool updateStreamlines		= false;
	bool updatePathlines		= false;
	bool updateDispersion		= true;
	bool updatefluctuation		= false;

	bool showStreamlines		= false;
	bool showPathlines			= false;
	bool showRaycasting			= false;
	bool showDispersion			= false;
	bool showFluctuationHeightfield = false;

	bool streamlineRendering	= true;
	bool pathlineRendering		= false;

	void drawSolverOptions();					// draw the solver option window
	void drawLog();								// draw Log window
	void drawLineRenderingOptions();			// draw options of stream/pathline rendering
	void drawRaycastingOptions();				// draw options of isosurface rendering (raycasting)
	void drawDispersionOptions();				// draw options of dispersion calculation
	void drawFluctuationHeightfieldOptions();	// draw options of heightfield of fluctuation 

	void render();								// render Dear ImGui

	// Log pointer
	char* log = new char[1000];
	float eyePos[3] = { 0,0,0 };
	float viewDir[3] = { 0,0,0 };
	float upDir[3] = { 0,0,0 };







};

