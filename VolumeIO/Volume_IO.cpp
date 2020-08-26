#include "Volume_IO.h"
#include "..//ErrorLogger/ErrorLogger.h"


// Read a velocity volume
bool VolumeIO::Volume_IO::readVolume(unsigned int idx)
{
	// Generate absolute path of the file
	
	this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";

	// Read volume into the buffer
	return Read();
}



std::vector<char>* VolumeIO::Volume_IO::getField_char()
{
	return &this->buffer;
}


float* VolumeIO::Volume_IO::getField_float()
{
	return field;
}

void VolumeIO::Volume_IO::release()
{

	this->buffer.clear();
	this->field = nullptr;

}

bool VolumeIO::Volume_IO::Read(std::streampos _begin, size_t size)
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
		//std::printf(std::string("Successfully Open File: " + fullName +  "\n").c_str());
	}

	//
	myFile.seekg(0, std::ios::beg);


	// return to starting position
	myFile.seekg(_begin, std::ios::beg);


	// resize it to fit the dataset
	(this->buffer).resize(size);

	//read file and store it into buffer 
	myFile.read(&(buffer.at(0)), size);


	// close the file
	myFile.close();

	this->field = reinterpret_cast<float*>(&(buffer.at(0)));

	return true;

}


bool VolumeIO::Volume_IO::Read()
{
	// define the istream
	std::ifstream myFile;

	myFile = std::ifstream(this->fullName, std::ios::binary);

	// check whether it can open the file
	if (!myFile.is_open())
	{
		std::string error_string = "Failed to open file : ";
		error_string += fullName;
		ErrorLogger::Log(error_string);
		return false;
	}
	else
	{
		std::printf(std::string("Successfully Open File: " + fullName + "\n").c_str());
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

void VolumeIO::Volume_IO::setFileName(std::string _fileName)
{
	this->m_fileName = _fileName;
}
void VolumeIO::Volume_IO::setFilePath(std::string _filePath)
{
	this->m_filePath = _filePath;
}


void VolumeIO::Volume_IO::Initialize(SolverOptions* _solverOptions)
{
	m_fileName = _solverOptions->fileName;
	m_filePath = _solverOptions->filePath;


}

void VolumeIO::Volume_IO::Initialize(RaycastingOptions* _raycastingOptions)
{
	m_fileName = _raycastingOptions->fileName;
	m_filePath = _raycastingOptions->filePath;

}


void VolumeIO::Volume_IO::Initialize(std::string _fileName, std::string _filePath)
{
	m_fileName = _fileName;
	m_filePath = _filePath;

}


bool VolumeIO::Volume_IO::isEmpty()
{
	if (field == nullptr)
		return true;
	return false;
}