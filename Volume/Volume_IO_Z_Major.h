#pragma once

#include "Volume_IO.h"
#include "../ErrorLogger/ErrorLogger.h"


class Volume_IO_Z_Major : public VolumeIO::Volume_IO
{


private:

	SolverOptions* m_solverOptions = nullptr;


public:


	virtual void Initialize(SolverOptions* _solverOptions) override
	{
		m_solverOptions = _solverOptions;

		m_fileName = _solverOptions->fileName;
		m_filePath = _solverOptions->filePath;
		this->initialized = true;
	}

	virtual void Initialize(FluctuationheightfieldOptions* _fluctuationOptions) override
	{
		m_fileName = _fluctuationOptions->fileName;
		m_filePath = _fluctuationOptions->filePath;

		this->initialized = true;
	}

	virtual bool readSliceXY(unsigned int idx, size_t x, size_t y) override
	{

		if (this->m_solverOptions == nullptr)
				return false;
		else
		{
			this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";
			std::streampos begin = 0;
			begin += x * m_solverOptions->gridSize[2] * (size_t) m_solverOptions->gridSize[1] * sizeof(float4);
			begin += y * m_solverOptions->gridSize[2] * sizeof(float4);

			size_t buffer_size = m_solverOptions->gridSize[2] * sizeof(float4);
			return this->Read(begin, buffer_size);
		}
	}

	virtual bool readVolumePlane(unsigned int idx, VolumeIO::readPlaneMode planeMode, size_t plane, size_t offset, size_t buffer_size)
	{
		// Generate absolute path of the file

		this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";

		std::streampos begin = 0;


		switch (static_cast<int>(planeMode))
		{
		case 0: // => YZ
			ErrorLogger::Log("Not implemented yet");


		case 1: // => ZX
			ErrorLogger::Log("Not implemented yet");

			break;

		case 2: // => XY


			begin = plane * offset;
			return this->Read(begin, buffer_size);

			break;
		}

		return false;
	}
};