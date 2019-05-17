
#include "BinaryWriter.h"
#include <assert.h>
#include <filesystem>

// define fs namespace for convenience
namespace fs = std::experimental::filesystem;


//########## getter functions ####################//

const char* BinaryWriter::getFileName() const {
	return this->m_fileName.c_str();
}
const char* BinaryWriter::getFilePath() const {
	return this->m_filePath.c_str();
}
//################################################//



//########## setter functions ####################//

void BinaryWriter::setFileName(const char* _fileName) {
	m_fileName = std::string(_fileName);
}
void BinaryWriter::setFilePath(const char* _filePath) {
	m_filePath = std::string(_filePath);
}
void BinaryWriter::setBuffer(char* _buffer) {
	this->m_buffer = _buffer;
}

void BinaryWriter::setBufferSize(std::size_t _buffer_size) {
	this->m_buffer_size = _buffer_size;
}

//################################################//


//############### Constructors ####################//

BinaryWriter::BinaryWriter()
{
	this->m_fileName = "";
	this->m_filePath = "";
}


BinaryWriter::BinaryWriter(const char* _fileName, const char* _filePath)
	: m_fileName(_fileName), m_filePath(_filePath)
{}


BinaryWriter::BinaryWriter(const char* _fileName, const char* _filePath, char* _buffer)
	: m_fileName(_fileName), m_filePath(_filePath), m_buffer(_buffer)
{}

BinaryWriter::BinaryWriter(const char* _fileName, const char* _filePath, char* _buffer, std::size_t _buffer_size)
	: m_fileName(_fileName), m_filePath(_filePath), m_buffer(_buffer), m_buffer_size(_buffer_size)
{}

BinaryWriter::BinaryWriter(const char* _fileName) {


	this->m_fileName = _fileName;

	// extract the current path
	fs::path path_currentPath = fs::current_path();
	this->m_filePath = path_currentPath.u8string();

	// add a backslash at the end of path
	this->m_filePath += "\\";

}
//################################################//

void BinaryWriter::write()
{
	// define the istream
	std::ofstream myFile;

	myFile = std::ofstream(m_filePath + m_fileName, std::ios::binary);

	// check whether the file is open
	assert(myFile.is_open());

	// write into the file
	myFile.write(&(this->m_buffer[0]), this->m_buffer_size);

	// close the file
	myFile.close();

}