#include "CudaSolver.h"

CUDASolver::CUDASolver()
{
	std::printf("A solver is created!\n");
}

// Initilize the solver
bool CUDASolver::Initialize
(
	SeedingPattern _seedingPattern,
	IntegrationMethod _integrationMethod,
	InterpolationMethod _interpolationMethod,
	SolverOptions _solverOptions
)
{
	this->m_seedingPattern = _seedingPattern;
	this->m_intergrationMehotd = _integrationMethod;
	this->m_interpolationMethod = _interpolationMethod;
	this->solverOptions = _solverOptions;

	return true;
}

bool CUDASolver::ReadField(std::vector<char>* p_vec_buffer, std::string fileName)
{
	// define the istream
	fileName = "D:\\git_projects\\Flow_Visualization\\Field1.bin";
	std::ifstream myFile;

	myFile = std::ifstream(fileName, std::ios::binary);

	// check whether it can open the file
	if (!myFile.is_open())
	{
		std::string error_string = "Failed to open file : ";
		error_string += fileName;
		ErrorLogger::Log(error_string);
		return false;
	}
	else
	{
		std::printf(std::string("Successfully Open File: " + fileName).c_str());
	}
	// get the starting position
	std::streampos start = myFile.tellg();

	// go to the end
	myFile.seekg(0, std::ios::end);

	// get the ending position
	std::streampos end = myFile.tellg();

	// return to starting position
	myFile.seekg(0, std::ios::beg);

	// size of the buffer
	const int buffer_size = static_cast<int>(end - start);

	// resize it to fit the dataset(MUST BE EDITED WHILE IT IS ABOVE THE RAM SIZE)
	(*p_vec_buffer).resize(buffer_size);

	//read file and store it into buffer 
	myFile.read(&(p_vec_buffer->at(0)), buffer_size);

	// close the file
	myFile.close();

	return true;
}

bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox)
{
	return true;
}


void CUDASolver::InitializeParticles(int& particle_count, float3& gridDiamters, SeedingPattern seedingPattern)
{
	this->h_particles = new Particle[particle_count];

	if (seedingPattern == SEED_RANDOM)
	{
		for (int i = 0; i < particle_count; i++)
		{
			h_particles[i].seedParticle(gridDiamters);
		}
	}
	//TO-DO:: Regular seeding

}
