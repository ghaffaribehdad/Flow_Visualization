#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "..//Options/DispresionOptions.h"
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

	


public:

	void setResources(Camera* _camera, Timer * _fpsTimer, RenderingOptions * _renderingOptions, SolverOptions * _solverOptions, RaycastingOptions * _raycastingOptions)
	{
		this->camera = _camera;
	
		this->fpsTimer = _fpsTimer;
	
		this->renderingOptions = _renderingOptions;
	
		this->solverOptions = _solverOptions;

		this->raycastingOptions = _raycastingOptions;
	}

	bool updateRaycasting		= false;
	bool updateLineRendering	= false;
	bool updateSeedBox			= false;
	bool updateVolumeBox		= false;
	bool updateStreamlines		= false;
	bool updatePathlines		= false;

	bool showStreamlines		= false;
	bool showPathlines			= false;
	bool showRaycasting			= false;


	bool streamlineRendering	= true;
	bool pathlineRendering		= false;

	void drawSolverOptions();			// draw the solver option window
	void drawLog();						// draw Log window
	void drawLineRenderingOptions();	// draw options of stream/pathline rendering
	void drawRaycastingOptions();		// draw options of isosurface rendering (raycasting)
	void drawDispersionOptions();		// draw options of dispersion calculation

	void render();						// render Dear ImGui

	// Log pointer
	char* log = new char[1000];







};

