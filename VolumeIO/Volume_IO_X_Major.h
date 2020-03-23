#pragma once

#include "Volume_IO.h"
#include "../ErrorLogger/ErrorLogger.h"

class Volume_IO_X_Major : public VolumeIO::Volume_IO
{

private:

	SolverOptions* m_solverOptions;


public:




	virtual void Initialize(SolverOptions* _solverOptions) override
	{
		m_solverOptions = _solverOptions;

		m_fileName = _solverOptions->fileName;
		m_filePath = _solverOptions->filePath;
	}

	virtual bool readVolumePlane(unsigned int idx, VolumeIO::readPlaneMode planeMode, size_t plane)
	{
		// Generate absolute path of the file

		this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";


		switch (planeMode)
		{
		case VolumeIO::readPlaneMode::YZ:
		{

			size_t planeSize_byte =
				(size_t)m_solverOptions->gridSize[1] *
				(size_t)m_solverOptions->gridSize[2] *
				m_solverOptions->channels *
				sizeof(float);

			size_t offset = plane * planeSize_byte;

			return this->Read(offset, planeSize_byte);
			break;
		}
		case VolumeIO::readPlaneMode::ZX:
		{

			size_t interval_x = sizeof(float) * m_solverOptions->channels * m_solverOptions->gridSize[0];
			size_t interval_z = interval_x * m_solverOptions->gridSize[1];
			size_t offset = interval_x * plane;

			for (int z = 0; z < m_solverOptions->gridSize[2]; z++)
			{
				for (int x = 0; x < m_solverOptions->gridSize[0]; x++)
				{
					this->Read(offset, sizeof(float) * m_solverOptions->channels);
					offset += interval_x;

				}

				offset += interval_z;
			}
			break;
		}
		case VolumeIO::readPlaneMode::XY:
		{
			size_t interval_y = sizeof(float) * m_solverOptions->channels * m_solverOptions->gridSize[1];
			size_t interval_z = interval_y * m_solverOptions->gridSize[0];
			size_t offset = interval_y * plane;

			for (int z = 0; z < m_solverOptions->gridSize[2]; z++)
			{
				for (int y = 0; y < m_solverOptions->gridSize[1]; y++)
				{
					this->Read(offset, sizeof(float) * m_solverOptions->channels);
					offset += interval_y;

				}

				offset += interval_z;
			}
			break;
		}
		}
		return false;

	}
};

