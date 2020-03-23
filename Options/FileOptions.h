#pragma once
#include <string>


struct FileOptions
{
	// Input file
	std::string r_fileName;
	std::string r_filePath;
	
	std::string w_fileName;
	std::string w_filePath;

	int currentIdx = 0;
	int firstIdx = 0;
	int lastIdx = 0;

	bool loaded = false;

	int gridSize[3] = { 0,0,0 };
};