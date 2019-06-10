#include "Volume_IO.h"


std::vector<char>* Volume_IO::readVolume(unsigned int idx)
{
	// Initialize the vector buffer
	this->p_buffer = new std::vector<char>;

	this->fullName = filePath + fileName + "idx" + "bin";

	// Read volume into the buffer
	this->Read(this->p_buffer);

	return p_buffer;
}


bool Volume_IO::Read(std::vector<char>* p_buffer)
{
	// define the istream
	std::ifstream myFile;

	myFile = std::ifstream(this->fullName, std::ios::binary);

	// check whether it can open the file
	if (!myFile.is_open())
	{
		std::string error_string = "Failed to open file : ";
		error_string += fileName;
		ErrorLogger::Log(error_string);
		return false;
	}
	else
	{
		std::printf(std::string("Successfully Open File: " + fileName).c_str());
	}
	// get the starting position
	std::streampos start = myFile.tellg();

	// go to the end
	myFile.seekg(0, std::ios::end);

	// get the ending position
	std::streampos end = myFile.tellg();

	// return to starting position
	myFile.seekg(0, std::ios::beg);

	// size of the buffer
	const int buffer_size = static_cast<int>(end - start);

	// resize it to fit the dataset(MUST BE EDITED WHILE IT IS ABOVE THE RAM SIZE)
	(*this->p_buffer).resize(buffer_size);

	//read file and store it into buffer 
	myFile.read(&(p_buffer->at(0)), buffer_size);

	// close the file
	myFile.close();

	return true;
}

void Volume_IO::setFileName(std::string _fileName) 
{
	this->fileName = _fileName;
}
void Volume_IO::setFilePath(std::string _filePath)
{
	this->filePath = _filePath;
}
void Volume_IO::setIndex(unsigned int _first, unsigned int _last)
{
	
}
