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
		std::string m_subPath = "";
		std::string fullName = "";
		bool isCompressed = false;
		bool isEnsemble = false;
		bool compressResourceInitialized = false;
		DecompressResources decompressResources;
		std::vector<unsigned int> index;
		std::vector<char> buffer;
		std::vector<char> buffer_compressed;
		float* p_field = nullptr;	// Stores a pointer to the field on the Host 
		float * dp_field = nullptr; 	// Stores a pointer to the field on the Device
		float* planeBuffer = nullptr;
		size_t buffer_size = 0;
		SolverOptions* solverOptions = nullptr;

		bool Read(std::streampos begin, size_t size);

		

	public:


		static bool file_check(std::string file)
		{
			std::ifstream myFile = std::ifstream(file.c_str(), std::ios::binary);

			// check whether it can open the file
			if (!myFile.is_open())
			{
				return false;
			}
			else
			{
				myFile.close();
				return true;
			}
		}


		Timer timer;
		// Setter and getter functions

		//void Initialize(SolverOptions* _solverOptions);
		void Initialize(FieldOptions * _fieldOptions);
		void Initialize( 
			std::string & _fileName,
			std::string & _filePath,
			bool & _compressed,
			bool & _compressResourceInitialized,
			std::size_t & _maxSize,
			int * gridSize
		);
		void InitializeBufferRealTime(SolverOptions* _solverOptions);

		void setFileName(std::string _fileName);
		void setFilePath(std::string _filePath);
		void setFileSubpath(std::string _fileSubpath);
		//bool isEmpty();

		
		bool readVolume(const unsigned int & idx , const int & member_idx = 0);
		bool readVolume_Compressed(const unsigned int & idx, const int3 & gridSize, const int & member_idx = 0);	// Generic: Read binary file with a certain index


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
			this->solverOptions = _solverOptions;
		}

		// Clear the vector
		void release();
		bool Read();
		bool Read_Compressed(int3 _gridSize);

	};

}

