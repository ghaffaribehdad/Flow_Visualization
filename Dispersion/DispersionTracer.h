#pragma once
#include "DispersionHelper.h"
#include "../BinaryStream/BinaryReader.h"
#include "../BinaryStream/BinaryWriter.h"
#include <string>

namespace Dispersion
{
	enum DispersionField
	{

		VECTOR_DIRECTION_MAGNITUDE_AVG_DST,
		VECTOR_DIRECTION_MAGNITUDE_TIME,
		VECTOR_LOCATION_MAGNITUDE_AVG_DST,
		VECTOR_LOCATION_MAGNITUDE_TIME
	};

	enum WriteMode
	{
		TEMPORARY,
		SAVE_TO_FILE
	};
}

class DispersionTracer
{
public:

	void initialize();
	void trace();
	void writeTemp();
	float * read();
	void read(std::string fileName, std::string filePath);
	void writeToFile(std::string fileName, std::string filePath);

private:

	Dispersion::DispersionField Mode;
	Dispersion::WriteMode writeMode;

	BinaryReader binaryReader;
	BinaryWriter binaryWriter;

	float* field;

};