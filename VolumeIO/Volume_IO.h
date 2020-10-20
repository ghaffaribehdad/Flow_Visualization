#pragma once

#include <vector>
#include <string>
#include <fstream>
#include "../Timer/Timer.h"
#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "Compression.h"

typedef unsigned int uint;

namespace VolumeIO
{
	enum readPlaneMode
	{
		YZ = 0,
		ZX,
		XY
	};


	class Volume_IO
	{
	protected:

		std::string m_fileName = "";
		std::string m_filePath = "";
		std::string fullName = "";

		std::vector<unsigned int> index;
		std::vector<char> buffer;
		float* p_field = nullptr;

		// Stores a pointer to the field on the GPU (device)
		float * dp_field = nullptr;
		size_t buffer_size = 0;
		size_t bufferSize = 0;
		float* planeBuffer = nullptr;
		SolverOptions* m_solverOptions = nullptr;

		bool Read(std::streampos begin, size_t size);
		
		DecompressResources decompressResources;

		

	public:

		Timer timer;
		// Setter and getter functions

		void Initialize(SolverOptions* _solverOptions);
		void Initialize(RaycastingOptions* _raycastingOptions);
		void Initialize(std::string _fileName, std::string _filePath);
		void InitializeRealTime(SolverOptions* _solverOptions);

		void setFileName(std::string _fileName);
		void setFilePath(std::string _filePath);
		bool isEmpty();

		


		bool readVolume(unsigned int idx);	// Generic: Read binary file with a certain index
		bool readVolume(unsigned int idx, SolverOptions * solverOptions);	// Generic: Read binary file with a certain index
		bool readVolume();					// Read binary file without index


		// Read a single plane of a volumetric file in binary
		virtual bool readVolumePlane(unsigned int idx, readPlaneMode planeMode, size_t plane) = 0;

		// Return a pointer to char vector
		std::vector<char>* getField_char();

		// Return a pointer to array of floats
		float* getField_float();
		float* getField_float_GPU();

		std::vector<char>::iterator getBegin()
		{
			return buffer.begin();
		}

		std::vector<char>::iterator getEnd()
		{
			return buffer.end();
		}
		const size_t & getBufferSize()
		{
			return buffer_size;
		}
		void setSolverOptions(SolverOptions * _solverOptions)
		{
			this->m_solverOptions = _solverOptions;
		}

		// Clear the vector
		void release();
		void releaseDecompressionResources();
		bool Read();
		bool Read_Compressed(SolverOptions * solverOptions);

	};

}

