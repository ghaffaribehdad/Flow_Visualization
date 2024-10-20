#pragma once

#include <map>
#include <string>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"

#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "../Options/DispresionOptions.h"
#include "../Options/SpaceTimeOptions.h"
#include "../Options/RenderingOptions.h"
#include "../Options/CrossSectionOptions.h"
#include "../Options/TurbulentMixingOptions.h"
#include "../Options/TimeSpace3DOptions.h"
#include "../Options/FieldOptions.h"
#include "../Options/VisitationOptions.h"
#include "../Options/pathSpaceTimeOptions.h"

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
	SpaceTimeOptions*				spaceTimeOptions;
	PathSpaceTimeOptions*			pathSpaceTimeOptions;
	TurbulentMixingOptions*			turbulentMixingOptions;
	TimeSpace3DOptions*				timeSpace3DOptions;
	VisitationOptions*				visitationOptions;

	int dataset[4] = { Dataset::Dataset::NONE ,Dataset::Dataset::NONE ,Dataset::Dataset::NONE ,Dataset::Dataset::NONE };
	Dataset::Dataset dataset_0 = Dataset::Dataset::NONE;
	Dataset::Dataset dataset_1 = Dataset::Dataset::NONE;
	Dataset::Dataset dataset_2 = Dataset::Dataset::NONE;
	Dataset::Dataset dataset_3 = Dataset::Dataset::NONE;
	Dataset::Dataset raycastyingDataset = Dataset::Dataset::NONE;

public:

	template<typename T>
	void static setArray(T* dest ,const T  &x, const T &y, const T &z)
	{
		dest[0] = x;
		dest[1] = y;
		dest[2] = z;
	}

	template<typename T>
	void static setArray(T* dest, T* src)
	{
		dest[0] = src[0];
		dest[1] = src[1];
		dest[2] = src[2];
	}

	void setResources
	(
		Camera					*_camera,
		Timer					*_fpsTimer, 
		RenderingOptions		*_renderingOptions,
		SolverOptions			*_solverOptions,
		RaycastingOptions		*_raycastingOptions,
		DispersionOptions		*_dispersionOptions,
		SpaceTimeOptions		*_spaceTimeOptions,
		CrossSectionOptions		*_crossSectionOptions,
		TurbulentMixingOptions	*_turbulentMixingOptions,
		TimeSpace3DOptions	 	*_timeSpace3DOptions,
		FieldOptions			*_fieldOptions,
		VisitationOptions		*_visitationOptions,
		PathSpaceTimeOptions	*_pathSpaceTimeOptions
	)
	{
		this->camera					= _camera;
		this->fpsTimer					= _fpsTimer;
		this->renderingOptions			= _renderingOptions;
		this->solverOptions				= _solverOptions;
		this->raycastingOptions			= _raycastingOptions;
		this->dispersionOptions			= _dispersionOptions;
		this->spaceTimeOptions			= _spaceTimeOptions;
		this->crossSectionOptions		= _crossSectionOptions;
		this->turbulentMixingOptions	= _turbulentMixingOptions;
		this->timeSpace3DOptions		= _timeSpace3DOptions;
		this->fieldOptions				= _fieldOptions;
		this->visitationOptions			= _visitationOptions;
		this->pathSpaceTimeOptions		= _pathSpaceTimeOptions;
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
	bool updateOIT				= false;
	bool updateShaders			= false;
	bool updateVisitationMap	= false;
	bool updatePathSpaceTime	= false;
	bool updateFile[4]			= { false,false,false,false };


	bool hideOptions = false;

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
	bool showVisitationMap				= false;
	bool showTimeSpaceField				= false;
	bool showPathSpaceTime				= false;

	bool releaseStreamlines = false;
	bool releaseStreaklines = false;
	bool releasePathlines = false;
	bool releaseRaycasting = false;
	bool releaseCrossSection = false;
	bool releaseDispersion = false;
	bool releaseFTLE = false;
	bool releaseTurbulentMixing = false;
	bool releaseFluctuationHeightfield = false;
	bool releaseVisitationMap = false;
	bool releaseTimeSpaceField = false;

	bool streamlineRendering	= true;
	bool pathlineRendering		= false;
	bool streamlineGenerating	= false;


	bool saveScreenshot = false;
	bool saved = false;


	void drawOptionWindows()
	{
		// first reinitialize values
		solverOptions->fileChanged = false;
		solverOptions->viewChanged = false;

		updateRaycasting = false;
		updateLineRendering = false;
		updateSeedBox = false;
		updateVolumeBox = false;
		updateStreamlines = false;
		updateStreaklines = false;
		updatePathlines = false;
		updateDispersion = true;
		updateFTLE = true;
		updatefluctuation = false;
		updateCrossSection = false;
		updateTurbulentMixing = false;
		updateTimeSpaceField = false;
		updateOIT = false;
		updateShaders = false;
		updateVisitationMap = false;
		updateFile[0] = false; updateFile[1] = false; updateFile[2] = false; updateFile[3] = false;
		updatePathSpaceTime = false;
		

		releaseStreamlines = false;
		releaseStreaklines = false;
		releasePathlines = false;
		releaseRaycasting = false;
		releaseCrossSection = false;
		releaseDispersion = false;
		releaseFTLE = false;
		releaseTurbulentMixing = false;
		releaseFluctuationHeightfield = false;
		releaseVisitationMap = false;
		releaseTimeSpaceField = false;


		this->drawImguiOptions();
		this->drawSolverOptions();	
		this->drawLog();								
		this->drawLineRenderingOptions();			
		this->drawRaycastingOptions();		
		this->drawVisitationMapOptions();
		this->drawFieldOptions();
		this->drawPathSpaceTime();
	}

	// render Dear ImGui
	void render();	

	// Log pointer
	char* log = new char[1000];
	float eyePos[3] = { 0,0,0 };
	float viewDir[3] = { 0,0,0 };
	float upDir[3] = { 0,0,0 };
	int nFields = 1;

	// Screenshot Options
	int screenshotRange = 1;
	int screenshotCounter = 0;
private:

	void drawImguiOptions();
	void drawSolverOptions();					// draw the solver option window
	void drawLog();								// draw Log window
	void drawLineRenderingOptions();			// draw options of stream/pathline rendering
	void drawFieldOptions();					// draw options of fields (datasets)
	void drawPathSpaceTime();				// draw options of surface options
	void drawRaycastingOptions();				// draw options of isosurface rendering (raycasting)
	void drawDispersionOptions();				// draw options of dispersion calculation
	void drawTimeSpaceOptions();				// draw options of heightfield of fluctuation 
	void drawCrossSectionOptions();				// draw options of CrossSection rendering
	void drawTimeSpaceField();
	void drawTurbulentMixingOptions();
	void drawDataset();
	void drawVisitationMapOptions();

	bool b_drawSolverOptions		= false;
	bool b_drawLog					= false;
	bool b_drawLineRenderingOptions = true;
	bool b_drawRaycastingOptions	= true;
	bool b_drawDispersionOptions	= false;
	bool b_drawTimeSpaceOptions		= false;
	bool b_drawVisitationOptions	= false;
	bool b_drawCrossSectionOptions	= false;
	bool b_drawTimeSpaceField		= false;
	bool b_drawDataset				= true;
};

