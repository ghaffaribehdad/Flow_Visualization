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
#include "../Options/TimeSpace3DOptions.h"
#include "../Options/FieldOptions.h"

#include "Camera.h"
#include "../Timer/Timer.h"




class RenderImGuiOptions
{



private:

	Camera*	camera;
	Timer*	fpsTimer;
	

	// Pointers to the Option structures
	SolverOptions*					solverOptions;
	FieldOptions *					fieldOptions;
	RenderingOptions*				renderingOptions;
	RaycastingOptions*				raycastingOptions;
	DispersionOptions*				dispersionOptions;
	CrossSectionOptions*			crossSectionOptions;
	SpaceTimeOptions*		fluctuationOptions;
	TurbulentMixingOptions*			turbulentMixingOptions;
	TimeSpace3DOptions*				timeSpace3DOptions;

	Dataset::Dataset dataset_0 = Dataset::Dataset::NONE;
	Dataset::Dataset dataset_1 = Dataset::Dataset::NONE;
	Dataset::Dataset raycastyingDataset = Dataset::Dataset::NONE;

public:

	template<typename T>
	void static setArray(T* source ,const T  &x, const T &y, const T &z)
	{
		source[0] = x;
		source[1] = y;
		source[2] = z;
	}

	void setResources
	(
		Camera* _camera,
		Timer * _fpsTimer, 
		RenderingOptions * _renderingOptions,
		SolverOptions * _solverOptions,
		RaycastingOptions * _raycastingOptions,
		DispersionOptions * _dispersionOptions,
		SpaceTimeOptions * _fluctuationheightfieldOptions,
		CrossSectionOptions * _crossSectionOptions,
		TurbulentMixingOptions* _turbulentMixingOptions,
		TimeSpace3DOptions * _timeSpace3DOptions,
		FieldOptions * _fieldOptions
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
		this->timeSpace3DOptions = _timeSpace3DOptions;
		this->fieldOptions = _fieldOptions;
	}

	bool updateRaycasting		= false;
	bool updateLineRendering	= false;
	bool updateSeedBox			= false;
	bool updateVolumeBox		= false;
	bool updateStreamlines		= false;
	bool updateStreaklines		= false;
	bool updatePathlines		= false;
	bool updateDispersion		= true;
	bool updateFTLE				= true;
	bool updatefluctuation		= false;
	bool updateCrossSection		= false;
	bool updateTurbulentMixing	= false;
	bool updateTimeSpaceField	= false;
	bool updateOIT	=			false;
	bool updateShaders = false;

	bool hideOptions = false;
	bool fileChanged = false;


	bool pauseRendering = false;
	
	bool showStreamlines				= false;
	bool showStreaklines				= false;
	bool showPathlines					= false;
	bool showRaycasting					= false;
	bool showCrossSection				= false;
	bool showDispersion					= false;
	bool showFTLE						= false;
	bool showTurbulentMixing			= false;
	bool showFluctuationHeightfield		= false;
	bool showTimeSpaceField				= false;


	bool releaseStreamlines = false;
	bool releaseStreaklines = false;
	bool releasePathlines = false;


	bool releaseTurbulentMixing = false;

	bool streamlineRendering	= true;
	bool pathlineRendering		= false;
	bool streamlineGenerating	= false;


	bool saveScreenshot = false;
	bool saved = false;


	void drawOptionWindows()
	{
		this->drawImguiOptions();
		this->drawSolverOptions();	
		this->drawLog();								
		this->drawLineRenderingOptions();			
		this->drawRaycastingOptions();				
		//this->drawDispersionOptions();				
		this->drawTimeSpaceOptions();	
		//this->drawCrossSectionOptions();			
		//this->drawTurbulentMixingOptions();
		//this->drawTimeSpaceField();
		this->drawDataset();
	}


	void render();								// render Dear ImGui

	// Log pointer
	char* log = new char[1000];
	float eyePos[3] = { 0,0,0 };
	float viewDir[3] = { 0,0,0 };
	float upDir[3] = { 0,0,0 };

	// Screenshot Options
	int screenshotRange = 1;
	int screenshotCounter = 0;
private:

	void drawImguiOptions();
	void drawSolverOptions();					// draw the solver option window
	void drawLog();								// draw Log window
	void drawLineRenderingOptions();			// draw options of stream/pathline rendering
	void drawRaycastingOptions();				// draw options of isosurface rendering (raycasting)
	void drawDispersionOptions();				// draw options of dispersion calculation
	void drawTimeSpaceOptions();				// draw options of heightfield of fluctuation 
	void drawCrossSectionOptions();				// draw options of CrossSection rendering
	void drawTimeSpaceField();
	void drawTurbulentMixingOptions();
	void drawDataset();

	bool b_drawSolverOptions		= true;
	bool b_drawLog					= false;
	bool b_drawLineRenderingOptions = true;
	bool b_drawRaycastingOptions	= false;
	bool b_drawDispersionOptions	= false;
	bool b_drawTimeSpaceOptions		= false;
	bool b_drawCrossSectionOptions	= false;
	bool b_drawTimeSpaceField		= false;
	bool b_drawDataset				= true;
};

