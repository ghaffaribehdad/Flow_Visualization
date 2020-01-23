#pragma once

#include "Volume_IO.h"
#include "../ErrorLogger/ErrorLogger.h"

class Volume_IO_X_Major : public volumeIO::Volume_IO
{

private:

	SolverOptions* m_solverOptions;

public:

	virtual bool readSliceXY(unsigned int idx, size_t x, size_t y) override
	{


		return false;
	}

	virtual bool readVolumePlane(unsigned int idx, volumeIO::readPlaneMode planeMode, size_t plane, size_t offset, size_t buffer_size)
	{
		// Generate absolute path of the file

		this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";

		std::streampos begin = 0;
		//size_t size = 0;


		switch (static_cast<int>(planeMode))
		{
		case 0: // => YZ
			begin = plane * offset;
			return this->Read(begin, buffer_size);


		case 1: // => ZX
			ErrorLogger::Log("Not implemented yet");
			break;

		case 2: // => XY
			ErrorLogger::Log("Not implemented yet");
			break;
		}

		return false;
	}
};

