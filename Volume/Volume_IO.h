#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "../ErrorLogger.h"
class Volume_IO
{
private:
	
	std::string fileName = "";
	std::string filePath = "";
	std::string fullName = "";
	std::vector<unsigned int> index;
	std::vector<char>* p_buffer;
public:
	
	void setFileName(std::string _fileName);
	void setFilePath(std::string _filePath);
	void setIndex(unsigned int _first, unsigned int _last);

	std::vector<char>* readVolume(unsigned int idx);

protected:

	bool Read(std::vector<char>* p_buffer);

};