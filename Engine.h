#pragma once

#include "WindowContainer.h"
#include "Timer.h"
#include "Cuda/StreamlineSolver.cuh"
#include "SolverOptions.h"



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

	bool InitializeStreamSolver();

private:
	Timer timer;

	StreamlineSolver streamlineSolver;

	//TO-DO:: implement Pathline Solver


};