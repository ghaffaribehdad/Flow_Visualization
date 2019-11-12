#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "../Options/SolverOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"

namespace volumeIO
{
	enum readPlaneMode
	{
		YZ = 0,
		ZX,
		XY
	};


	class Volume_IO
	{
	private:
	
		std::string m_fileName = "";
		std::string m_filePath = "";
		std::string fullName = "";
		std::vector<unsigned int> index;
		std::vector<char> buffer;
		float * field = nullptr;
		SolverOptions * m_solverOptions;
	public:
	
		void Initialize(SolverOptions * _solverOption);
		void Initialize(FluctuationheightfieldOptions * _fluctuationOptions);
		void setFileName(std::string _fileName);
		void setFilePath(std::string _filePath);

		bool isEmpty();
		bool readVolume(unsigned int idx);
		bool readVolumePlane(unsigned int idx, readPlaneMode planeMode, size_t plane, size_t offset ,size_t buffer_size);

		std::vector<char>* flushBuffer();
		float* flushBuffer_float();

		void release();
	protected:

		bool Read();
		bool Read(std::streampos begin, size_t size);

	};

}