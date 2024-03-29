#pragma once

#include "Volume_IO.h"
#include "../ErrorLogger/ErrorLogger.h"


class Volume_IO_Z_Major : public VolumeIO::Volume_IO
{


public:

	virtual bool readVolumePlane(unsigned int idx, VolumeIO::readPlaneMode planeMode, size_t plane)
	{
		// Solver Options is essential
		if (this->solverOptions == nullptr)
			return false;

		// Generate absolute path of the file
		this->fullName = m_filePath + m_fileName + std::to_string(idx) + ".bin";

		switch (planeMode)
		{
		case VolumeIO::readPlaneMode::YZ:
		{
			planeBuffer = new float[(size_t)solverOptions->gridSize[1] * (size_t)solverOptions->gridSize[2] * solverOptions->channels];

			size_t interval_x = sizeof(float) * solverOptions->channels;
			size_t interval_y = interval_x * solverOptions->gridSize[0];
			size_t interval_z = interval_y * solverOptions->gridSize[1];


			size_t offset = interval_y * plane;

			size_t index = 0;

			for (int z = 0; z < solverOptions->gridSize[2]; z++)
			{
				for (int y = 0; y < solverOptions->gridSize[1]; y++)
				{
					this->Read(offset, sizeof(float) * solverOptions->channels);
					offset += interval_y;

					planeBuffer[index] = p_field[0];
					planeBuffer[index + (size_t)1] = p_field[1];
					planeBuffer[index + (size_t)2] = p_field[2];
					planeBuffer[index + (size_t)3] = p_field[3];

					index += solverOptions->channels;

				}

				offset += interval_z;
			}
			break;

		}
		case VolumeIO::readPlaneMode::ZX:
		{

			planeBuffer = new float[(size_t)solverOptions->gridSize[0] * (size_t)solverOptions->gridSize[2] * solverOptions->channels];

			size_t interval_x = sizeof(float) * solverOptions->channels;
			size_t interval_y = interval_x * solverOptions->gridSize[0];
			size_t interval_z = interval_y * solverOptions->gridSize[1];
			size_t offset = interval_y * plane;

			size_t index = 0;
			for (int z = 0; z < solverOptions->gridSize[2]; z++)
			{
				std::printf(std::string("Successfully Read Plane: " + std::to_string(z) + "/" + std::to_string(solverOptions->gridSize[2]) + std::string("\n")).c_str());

				for (int x = 0; x < solverOptions->gridSize[0]; x++)
				{
					this->Read(offset, sizeof(float) * solverOptions->channels);
					offset += interval_x;

					planeBuffer[index + (size_t)0] = p_field[0];
					planeBuffer[index + (size_t)1] = p_field[1];
					planeBuffer[index + (size_t)2] = p_field[2];
					planeBuffer[index + (size_t)3] = p_field[3];

					index += solverOptions->channels;
				}

				offset += interval_z;
			}
			break;
		}

		case VolumeIO::readPlaneMode::XY:
		{
			size_t planeSize_byte =
				(size_t)solverOptions->gridSize[0] *
				(size_t)solverOptions->gridSize[1] *
				solverOptions->channels *
				sizeof(float);

			size_t offset = plane * planeSize_byte;

			this->Read(offset, planeSize_byte);
			planeBuffer = p_field;
			break;
		}

		}

		return false;
	}
};