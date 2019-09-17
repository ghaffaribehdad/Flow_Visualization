#pragma once

#include "../Window/WindowContainer.h"
#include "../Timer/Timer.h"
#include "../Cuda/StreamlineSolver.h"
#include "../Cuda/PathlineSolver.h"
#include <ScreenGrab.h>




class Engine : WindowContainer
{
public:

	
	// Initilize window, grphics and CudaSolver
	bool Initialize(HINSTANCE hInstance, std::string window_title, std::string window_class, int width, int height);
	// To process Messages
	bool ProcessMessages();

	// Update base on the new messages
	void Update();

	void RenderFrame();

	void release();


private:
	Timer timer;




};