#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "../Options/SolverOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"

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
		float * field = nullptr;


		bool initialized = false;

	public:
	

		// Setter and getter functions
		virtual void Initialize(SolverOptions * _solverOption);
		virtual void Initialize(FluctuationheightfieldOptions * _fluctuationOptions);
		void Initialize(std::string _fileName , std::string _filePath);

		void setFileName(std::string _fileName);
		void setFilePath(std::string _filePath);

		bool isEmpty();


		bool readVolume(unsigned int idx);
		virtual bool readVolumePlane(unsigned int idx, readPlaneMode planeMode, size_t plane, size_t offset, size_t buffer_size) = 0;
		virtual bool readSliceXY(unsigned int idx, size_t x, size_t y) = 0;

	
		std::vector<char>* getField_char();
		float* getField_float();

		// Clear the vector
		void release();
		bool Read();

	protected:

		bool Read(std::streampos begin, size_t size);

	};

}