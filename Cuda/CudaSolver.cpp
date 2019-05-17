

#include "CudaSolver.h"

CUDASolver::CUDASolver()
{
	std::printf("A solver is created!\n");
}

bool CUDASolver::Initialize(SolveMode _solveMode, SeedingPattern _seedingPattern, InterpolationMethod, unsigned int _InitialTimestep, unsigned _intFinalTimestep)
{
	if (_solveMode == STREAMLINE)
	{


	}

	if (_solveMode == PATHLINE)
	{
		ErrorLogger::Log("PathLine is not implemented yet!!");
		return false;
	}
	return true;
}