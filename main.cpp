
#include <Windows.h>
#include "Engine\Engine.h"
#include <random>
#include <io.h>
#include <fcntl.h>

int APIENTRY wWinMain(
	_In_ HINSTANCE hInstance,\
	_In_opt_ HINSTANCE hPrevInstance,\
	_In_ LPWSTR lpCmdLine,\
	_In_ int nCmdShow)
{


	// Source: https://justcheckingonall.wordpress.com/2008/08/29/console-window-win32-app/

	AllocConsole();

	HANDLE handle_out = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long)handle_out, _O_TEXT);
	FILE* hf_out = _fdopen(hCrt, "w");
	setvbuf(hf_out, NULL, _IONBF, 1);
	*stdout = *hf_out;

	HANDLE handle_in = GetStdHandle(STD_INPUT_HANDLE);
	hCrt = _open_osfhandle((long)handle_in, _O_TEXT);
	FILE* hf_in = _fdopen(hCrt, "r");
	setvbuf(hf_in, NULL, _IONBF, 128);
	*stdin = *hf_in;


	// define the engine
	Engine engine;


	HRESULT hr = CoInitialize(NULL);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Coinitialize!");
		return -1;
	}

	// initialize the engine
	if (engine.Initialize(hInstance, "Flow Visualization", "MyWindowClass", 1024, 764))
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
