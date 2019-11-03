#include "Volume_IO.h"
#include "..//ErrorLogger/ErrorLogger.h"

bool Volume_IO::readVolume(unsigned int idx)
{
	// Generate absolute path of the file
	
	this->fullName = filePath + fileName + std::to_string(index.at(idx-1)) + ".bin";

	// Read volume into the buffer
	return Read();
}

std::vector<char>* Volume_IO::flushBuffer()
{
	return &this->buffer;
}


float* Volume_IO::flushBuffer_float()
{
	return field;
}

void Volume_IO::release()
{

	this->buffer.clear();
	this->field = nullptr;

}

bool Volume_IO::Read()
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

	// resize it to fit the dataset
	(this->buffer).resize(buffer_size);

	//read file and store it into buffer 
	myFile.read(&(buffer.at(0)), buffer_size);

	// close the file
	myFile.close();


	this->field = reinterpret_cast<float*>(&(buffer.at(0)));

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


void Volume_IO::Initialize(SolverOptions* solverOptions)
{
	fileName = solverOptions->fileName;
	filePath = solverOptions->filePath;
	this->index.resize(solverOptions->lastIdx - solverOptions->firstIdx+1);

	for (int i = solverOptions->firstIdx; i <= solverOptions->lastIdx; i++)
	{
		index[i] = solverOptions->firstIdx;
	}
}

bool Volume_IO::isEmpty()
{
	if (field == nullptr)
		return true;
	return false;
}