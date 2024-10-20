
#include <Windows.h>
#include "Engine\Engine.h"
#include <random>
#include <io.h>
#include <fcntl.h>
#include <iostream>

int APIENTRY wWinMain(
	_In_ HINSTANCE hInstance,\
	_In_opt_ HINSTANCE hPrevInstance,\
	_In_ LPWSTR lpCmdLine,\
	_In_ int nCmdShow)
{

	// https://stackoverflow.com/questions/15543571/allocconsole-not-displaying-cout
	AllocConsole();
	freopen("CONOUT$", "w", stdout);

//#pragma enable_d3d11_debug_symbols

	// define the engine
	Engine engine;


	HRESULT hr = CoInitialize(NULL);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Coinitialize!");
		return -1;
	}

	// initialize the engine
	if (engine.Initialize(hInstance, "Flow Visualization", "MyWindowClass", 1024, 950))
	{
		while (engine.ProcessMessages() == true)
		{
			engine.Update();
			engine.RenderFrame();
		}

		engine.release();
	}
	
	return 0;
}
