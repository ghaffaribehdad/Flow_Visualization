// act as include guard
#pragma once

#include <string>

class StringConverter
{
public:
	// convert a std string to std wide string and return it
	static std::wstring StringToWide(std::string str);
};