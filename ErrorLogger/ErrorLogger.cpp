#include "ErrorLogger.h"
#include <comdef.h> //com error data type


void ErrorLogger::Log(std::string message)
{
	std::string error_message = "Error: " + message;
	MessageBoxA(NULL, error_message.c_str(), "Error", MB_ICONERROR);
}

void ErrorLogger::Log(HRESULT hr, std::string message)
{
	_com_error error(hr);
	//error.ErrorMessage() //retrieve information about the error
	std::wstring error_message = L"Error: " + StringConverter::StringToWide(message) + L"\n"+ error.ErrorMessage();
	MessageBoxW(NULL, error_message.c_str(), L"Error", MB_ICONERROR);
}

void ErrorLogger::Log(HRESULT hr, std::wstring message)
{
	_com_error error(hr);
	//error.ErrorMessage() //retrieve information about the error
	std::wstring error_message = L"Error: " + message + L"\n" + error.ErrorMessage();
	MessageBoxW(NULL, error_message.c_str(), L"Error", MB_ICONERROR);
}

bool ErrorLogger::Log(cudaError_t code, const char* file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		std::string error_message = "GPUassert: ";
		error_message += cudaGetErrorString(code);
		error_message += ", ";
		error_message += file;
		error_message += ", ";
		error_message += std::to_string(line);
#ifdef RELEASE
		std::printf("%s\n", error_message.c_str());
		MessageBoxW(NULL, StringConverter::StringToWide(error_message).c_str(), L"Error", MB_ICONERROR);
#endif // RELEASE
#ifdef DEBUG
		MessageBoxW(NULL, StringConverter::StringToWide(error_message).c_str(), L"Error", MB_ICONERROR);
#endif
		return false;
	}
	else
	{
		return true;
	}
}

