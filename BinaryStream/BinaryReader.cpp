
#include "BinaryReader.h"
#include <fstream>
#include <filesystem>
#include <cassert>

// define fs namespace for convenience
namespace fs = std::experimental::filesystem;

// keep track fo instantiation  
unsigned int BinaryReader::s_instances = 0;


// constructor
BinaryReader::BinaryReader(const char* _fileName) 
{

	//increment number of instances
	s_instances += 1;

	this->m_fileName = _fileName;

	// extract the current path
	fs::path path_currentPath = fs::current_path();
	this->m_filePath = path_currentPath.u8string();

	// add a backslash at the end of path
	this->m_filePath += "\\";

}

// constructor
BinaryReader::BinaryReader(const char* _fileName, const char* _filePath)
	: m_fileName(_fileName), m_filePath(_filePath)
{}

// getter functions
const char* BinaryReader::getfileName() const 
{
	return m_fileName.c_str();
}
const char* BinaryReader::getfilePath() const 
{
	return m_filePath.c_str();
}

// setter
void BinaryReader::setfileName(const char* _fileName) 
{
	m_fileName = std::string(_fileName);
}
void BinaryReader::setfilePath(const char* _filePath) 
{
	m_filePath = std::string(_filePath);
}



void BinaryReader::read(std::vector<char>* p_vec_buffer) 
{
	// define the istream
	std::ifstream myFile;

	myFile = std::ifstream(m_filePath + m_fileName, std::ios::binary);

	// check whether the file is open
	assert(myFile.is_open());

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
	(*p_vec_buffer).resize(buffer_size);

	//read file and store it into buffer 
	myFile.read(&(p_vec_buffer->at(0)), buffer_size);

	// close the file
	myFile.close();
}



std::vector<char>* BinaryReader::readFile()
{
	read(&(this->m_vec_buffer));
	std::vector<char>* buffer = this->flush_buffer();
	return buffer;
}

std::vector<char>* BinaryReader::flush_buffer() 
{
	return & m_vec_buffer;
}

void BinaryReader::clean_buffer() 
{
	this->m_vec_buffer.clear();
}

BinaryReader::BinaryReader()
{
	this->m_fileName = "";
	this->m_filePath = "";
	this->m_vec_buffer = {};
}

void BinaryReader::setVecBuffer(const std::vector<char> * _vec_buffer)
{
	this->m_vec_buffer = *_vec_buffer;
}