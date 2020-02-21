#pragma once

#include <map>
#include <string>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"

#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "../Options/DispresionOptions.h"
#include "../Options/fluctuationheightfieldOptions.h"
#include "../Options/RenderingOptions.h"
#include "../Options/CrossSectionOptions.h"
#include "../Options/TurbulentMixingOptions.h"

#include "Camera.h"
#include "../Timer/Timer.h"




class RenderImGuiOptions
{



private:

	Camera*	camera;
	Timer*	fpsTimer;

	// Pointers to the Option structures
	SolverOptions*					solverOptions;
	RenderingOptions*				renderingOptions;
	RaycastingOptions*				raycastingOptions;
	DispersionOptions*				dispersionOptions;
	CrossSectionOptions*			crossSectionOptions;
	FluctuationheightfieldOptions*	fluctuationOptions;
	TurbulentMixingOptions*			turbulentMixingOptions;

	// background color
	float bgColor[3] = { 0,0,0 };

public:

	void setResources
	(
		Camera* _camera,
		Timer * _fpsTimer, 
		RenderingOptions * _renderingOptions,
		SolverOptions * _solverOptions,
		RaycastingOptions * _raycastingOptions,
		DispersionOptions * _dispersionOptions,
		FluctuationheightfieldOptions * _fluctuationheightfieldOptions,
		CrossSectionOptions * _crossSectionOptions,
		TurbulentMixingOptions* _turbulentMixingOptions
	)
	{
		this->camera	= _camera;
		this->fpsTimer	= _fpsTimer;
	
		this->renderingOptions		= _renderingOptions;
		this->solverOptions			= _solverOptions;
		this->raycastingOptions		= _raycastingOptions;
		this->dispersionOptions		= _dispersionOptions;
		this->fluctuationOptions	= _fluctuationheightfieldOptions;
		this->crossSectionOptions	= _crossSectionOptions;
		this->turbulentMixingOptions = _turbulentMixingOptions;
	}

	bool updateRaycasting		= false;
	bool updateLineRendering	= false;
	bool updateSeedBox			= false;
	bool updateVolumeBox		= false;
	bool updateStreamlines		= false;
	bool updatePathlines		= false;
	bool updateDispersion		= true;
	bool updateFTLE				= true;
	bool updatefluctuation		= false;
	bool updateCrossSection		= false;
	bool updateTurbulentMixing	= false;

	
	
	bool showStreamlines		= false;
	bool showPathlines			= false;
	bool showRaycasting			= false;
	bool showCrossSection		= false;
	bool showDispersion			= false;
	bool showFTLE				= false;
	bool showTurbulentMixing	= false;
	bool showFluctuationHeightfield = false;

	bool releaseTurbulentMixing = false;


	bool streamlineRendering	= true;
	bool pathlineRendering		= false;
	bool streamlineGenerating	= false;



	void drawOptionWindows()
	{
		this->drawSolverOptions();	
		this->drawLog();								
		this->drawLineRenderingOptions();			
		this->drawRaycastingOptions();				
		this->drawDispersionOptions();				
		this->drawFluctuationHeightfieldOptions();	
		this->drawCrossSectionOptions();			
		this->drawTurbulentMixingOptions();
	}


	void render();								// render Dear ImGui

	// Log pointer
	char* log = new char[1000];
	float eyePos[3] = { 0,0,0 };
	float viewDir[3] = { 0,0,0 };
	float upDir[3] = { 0,0,0 };


private:


	void drawSolverOptions();					// draw the solver option window
	void drawLog();								// draw Log window
	void drawLineRenderingOptions();			// draw options of stream/pathline rendering
	void drawRaycastingOptions();				// draw options of isosurface rendering (raycasting)
	void drawDispersionOptions();				// draw options of dispersion calculation
	void drawFluctuationHeightfieldOptions();	// draw options of heightfield of fluctuation 
	void drawCrossSectionOptions();				// draw options of CrossSection rendering
	void drawTurbulentMixingOptions();


};

