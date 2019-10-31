#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "../Options/SolverOptions.h"

class Volume_IO
{
private:
	
	std::string fileName = "";
	std::string filePath = "";
	std::string fullName = "";
	std::vector<unsigned int> index;
	std::vector<char> buffer;
	float * field = nullptr;
public:
	
	void Initialize(SolverOptions * solverOption);
	void setFileName(std::string _fileName);
	void setFilePath(std::string _filePath);

	bool isEmpty();
	bool readVolume(unsigned int idx);

	std::vector<char>* flushBuffer();
	float* flushBuffer_float();

	void release();
protected:

	bool Read();

};