#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "..//SolverOptions.h"
#include <string>
#include "RenderingOptions.h"
#include "Camera.h"
#include "..//Timer.h"
#include <map>

class RenderImGuiOptions
{



private:

	Camera* camera;
	Timer* fpsTimer;
	SolverOptions* solverOptions;
	RenderingOptions* renderingOptions;

	


public:

	void setResources(Camera* _camera, Timer * _fpsTimer, RenderingOptions * _renderingOptions, SolverOptions * _solverOptions)
	{
		this->camera = _camera;
	
		this->fpsTimer = _fpsTimer;
	
		this->renderingOptions = _renderingOptions;
	
		this->solverOptions = _solverOptions;
	}

	bool updateLineRendering = false;

	void drawSolverOptions(); // draw the solver option window
	void drawLog();										// draw Log window
	void drawLineRenderingOptions();
	void render(); // renders the imGui drawings

	// Log pointer
	char* log = new char[1000];







};

