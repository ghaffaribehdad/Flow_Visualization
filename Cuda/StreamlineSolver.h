#pragma once
#include "CudaSolver.h"
class StreamlineSolver : CUDASolver
{


private:
	std::string datasetPath;
	std::string datasetFileName;
	unsigned int m_initialTimestep = 0;
	unsigned int m_finalTimestep = 0;
	XMINT3 dimension;

};