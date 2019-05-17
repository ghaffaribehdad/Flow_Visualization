#pragma once

#include "WindowContainer.h"
#include "Timer.h"
#include "Cuda/CudaSolver.h"
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


private:
	Timer timer;
	CUDASolver Streamline;


};