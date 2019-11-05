#include "Volume_IO.h"
#include "..//ErrorLogger/ErrorLogger.h"

bool volumeIO::Volume_IO::readVolume(unsigned int idx)
{
	// Generate absolute path of the file
	
	this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";

	// Read volume into the buffer
	return Read();
}


bool volumeIO::Volume_IO::readVolumePlane(unsigned int idx, readPlaneMode planeMode, int plane)
{
	// Generate absolute path of the file

	this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";
	
	std::streampos begin = 0;
	size_t size = 0;

	switch (static_cast<int>(planeMode))
	{
	case 0: // => YZ
		size = (size_t)m_solverOptions->gridSize[1] * (size_t)m_solverOptions->gridSize[2] * sizeof(float4);
		begin = plane * size;
		return this->Read(begin, size);
		break;
		
	case 1: // => ZX
		size = (size_t)m_solverOptions->gridSize[0] * (size_t)m_solverOptions->gridSize[2] * sizeof(float4);
		ErrorLogger::Log("Not implemented yet");

		break;

	case 2: // => XY
		size = (size_t)m_solverOptions->gridSize[0] * (size_t)m_solverOptions->gridSize[1] * sizeof(float4);
		ErrorLogger::Log("Not implemented yet");
		break;
	}

	
}

std::vector<char>* volumeIO::Volume_IO::flushBuffer()
{
	return &this->buffer;
}


float* volumeIO::Volume_IO::flushBuffer_float()
{
	return field;
}

void volumeIO::Volume_IO::release()
{

	this->buffer.clear();
	this->field = nullptr;

}

bool volumeIO::Volume_IO::Read(std::streampos begin, size_t size)
{


	// define the ifstream
	std::ifstream myFile;

	myFile = std::ifstream(this->fullName, std::ios::binary);

	// check whether it can open the file
	if (!myFile.is_open())
	{
		std::string error_string = "Failed to open file : ";
		error_string += m_fileName;
		ErrorLogger::Log(error_string);
		return false;
	}
	else
	{
		std::printf(std::string("Successfully Open File: " + m_fileName).c_str());
	}

	// return to starting position
	myFile.seekg(0, myFile.beg);

	// return to starting position
	myFile.seekg(begin);

	// size of the buffer
	size_t buffer_size = static_cast<size_t>(size);


	// resize it to fit the dataset
	(this->buffer).resize(buffer_size);

	//read file and store it into buffer 
	myFile.read(&(buffer.at(0)), buffer_size);

	// close the file
	myFile.close();

	this->field = reinterpret_cast<float*>(&(buffer.at(0)));

	return true;

}


bool volumeIO::Volume_IO::Read()
{
	// define the istream
	std::ifstream myFile;

	myFile = std::ifstream(this->fullName, std::ios::binary);

	// check whether it can open the file
	if (!myFile.is_open())
	{
		std::string error_string = "Failed to open file : ";
		error_string += m_fileName;
		ErrorLogger::Log(error_string);
		return false;
	}
	else
	{
		std::printf(std::string("Successfully Open File: " + m_fileName).c_str());
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

void volumeIO::Volume_IO::setFileName(std::string _fileName)
{
	this->m_fileName = _fileName;
}
void volumeIO::Volume_IO::setFilePath(std::string _filePath)
{
	this->m_filePath = _filePath;
}


void volumeIO::Volume_IO::Initialize(SolverOptions* _solverOptions)
{
	m_fileName = _solverOptions->fileName;
	m_filePath = _solverOptions->filePath;
	m_solverOptions = _solverOptions;
}

bool volumeIO::Volume_IO::isEmpty()
{
	if (field == nullptr)
		return true;
	return false;
}