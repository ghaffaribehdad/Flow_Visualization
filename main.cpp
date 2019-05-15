#include <Windows.h>
#include "Engine.h"

// Create Error Logging for GPU
#define gpuErrchk(ans) { ErrorLogger::Log((ans), __FILE__, __LINE__); }



int APIENTRY wWinMain(
	_In_ HINSTANCE hInstance,\
	_In_opt_ HINSTANCE hPrevInstance,\
	_In_ LPWSTR lpCmdLine,\
	_In_ int nCmdShow)
{
	// define the engine
	Engine engine;
	
	HRESULT hr = CoInitialize(NULL);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Coinitialize!");
		return -1;
	}

	// initialize the engine
	if (engine.Initialize(hInstance, "Title", "MyWindowClass", 1024, 748))
	{
		while (engine.ProcessMessages() == true)
		{
			engine.Update();
			engine.RenderFrame();
		}
	}
	
	return 0;
}
