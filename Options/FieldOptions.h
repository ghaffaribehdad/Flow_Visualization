#pragma once
#include <string>

struct FieldOptions
{
	std::string fileName;
	std::string filePath;
	int gridSize[3] = { 0,0,0 };
	float gridDiameter[3] = { 0,0,0 };
	bool compressed = false;
	bool periodic = false;
	int firstIdx = 0;
	int lastIdx = 0;
	float dt = 0;
	size_t fileSizeMaxByte = 0;
};