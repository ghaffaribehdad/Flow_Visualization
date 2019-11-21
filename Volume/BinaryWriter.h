#pragma once


#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>


class BinaryWriter
{
private:
	std::string m_fileName;
	std::string m_filePath;

	char* m_buffer;

	// buffer size in byte
	std::size_t m_buffer_size;

	// by default the mode is set to "b" -> "binary"
	char mod = 'b';

public:
	BinaryWriter();
	BinaryWriter(const char* _fileName);
	BinaryWriter(const char* _fileName, const char* _filePath);
	BinaryWriter(const char* _fileName, const char* _filePath, char* _buffer);
	BinaryWriter(const char* _fileName, const char* _filePath, char* _buffer, std::size_t _buffer_size);



	const char* getFileName() const;
	const char* getFilePath() const;

	void setFileName(const char* _fileName);
	void setFileName(std::string _fileName);
	void setFilePath(const char* _filePath);
	void setBuffer(char* _buffer);

	void setBufferSize(std::size_t _buffer_size);


	bool write();
	bool append();



};

