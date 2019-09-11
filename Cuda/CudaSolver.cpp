#include "CudaSolver.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//ErrorLogger/ErrorLogger.h"


CUDASolver::CUDASolver()
{
	std::printf("A solver is created!\n");
}

// Initilize the solver
bool CUDASolver::Initialize(SolverOptions * _solverOptions)
{
	this->solverOptions = _solverOptions;
	this->InitializeCUDA();
	
	return true;
}


bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox)
{
	return true;
}


bool CUDASolver::FinalizeCUDA()
{
	gpuErrchk(cudaGraphicsUnmapResources(1,	&this->cudaGraphics	));

	gpuErrchk(cudaGraphicsUnregisterResource(this->cudaGraphics));

	return true;
}

bool CUDASolver::InitializeCUDA()
{
	// Get number of CUDA-Enable devices
	int device;
	gpuErrchk(cudaD3D11GetDevice(&device,solverOptions->p_Adapter));

	// Get properties of the Best(usually at slot 0) card
	gpuErrchk(cudaGetDeviceProperties(&this->cuda_device_prop, 0));

	// Register Vertex Buffer to map it
	gpuErrchk(cudaGraphicsD3D11RegisterResource(
		&this->cudaGraphics,
		this->solverOptions->p_vertexBuffer,
		cudaGraphicsRegisterFlagsNone));

	// Map Vertex Buffer
	gpuErrchk(cudaGraphicsMapResources(
		1,
		&this->cudaGraphics
		));

	// Get Mapped pointer
	size_t size = static_cast<size_t>(solverOptions->lines_count)* static_cast<size_t>(solverOptions->lineLength)*sizeof(Vertex);

	gpuErrchk(cudaGraphicsResourceGetMappedPointer(
		&p_VertexBuffer,
		&size,
		this->cudaGraphics
	));

	return true;
}

__host__ float* CUDASolver::InitializeVelocityField(int ID)
{
	this->volume_IO.readVolume(ID);
	std::vector<char>* p_vec_buffer = volume_IO.flushBuffer();
	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));


	return reinterpret_cast<float*>(p_vec_buffer_temp);
}



void CUDASolver::InitializeParticles(SeedingPattern seedingPattern)
{
	// Create an array of particles
	this->h_Particles = new Particle[solverOptions->lines_count];


	switch (seedingPattern)
	{
		case SeedingPattern::SEED_RANDOM:
		{
			// Seed Particles Randomly according to the grid diameters
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->h_Particles[i].seedParticle(solverOptions->gridDiameter,solverOptions->seedBox, solverOptions->seedBoxPos);
			}
			break;
		}

		case SeedingPattern::SEED_GRIDPOINTS:
		{
			//seedParticleGridPoints()
			break;
		}
		
		case SeedingPattern::SEED_FILE:
		{
			break;
		}


	}

	size_t Particles_byte = sizeof(Particle) * solverOptions->lines_count;

	// Upload Velocity Filled to GPU 

	gpuErrchk(cudaMalloc((void**) &this->d_Particles, Particles_byte));

	gpuErrchk(cudaMemcpy(this->d_Particles, this->h_Particles, Particles_byte, cudaMemcpyHostToDevice));

	delete[] this->h_Particles;
}

