#include "CudaSolver.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "../Particle/ParticleHelperFunctions.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "../Timer/Timer.h"
CUDASolver::CUDASolver()
{
	//std::printf("A solver is created!\n");
}

// Initilize the solver
bool CUDASolver::Initialize(SolverOptions * _solverOptions)
{
	this->solverOptions = _solverOptions;
	this->InitializeCUDA();
	this->volume_IO.Initialize(_solverOptions);
	
	return true;
}

bool CUDASolver::Reinitialize()
{
	this->InitializeCUDA();

	return true;
}

void CUDASolver::releaseVolumeIO()
{
	this->volume_IO.release();
}


bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox)
{
	return true;
}


bool CUDASolver::FinalizeCUDA()
{
	gpuErrchk(cudaGraphicsUnmapResources(1, &this->cudaGraphics));

	gpuErrchk(cudaGraphicsUnregisterResource(this->cudaGraphics));
	
	return true;
}

bool CUDASolver::InitializeCUDA()
{

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
				seedParticleRandom(h_Particles, solverOptions);
			}
			break;
		}

		case SeedingPattern::SEED_GRIDPOINTS:
		{
			// Create an array of particles

			seedParticleGridPoints(this->h_Particles, solverOptions);
			break;
		}

		case SeedingPattern::SEED_TILTED_PLANE:
		{
			float3 gridDiamter = Array2Float3(solverOptions->gridDiameter);
			seedParticle_tiltedPlane
			(
				this->h_Particles,
				gridDiamter,
				make_int2(solverOptions->gridSize_2D[0], solverOptions->gridSize_2D[1]),
				solverOptions->seedWallNormalDist,
				solverOptions->tilt_deg
			);
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

void CUDASolver::loadTexture
(
	SolverOptions * solverOptions,
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX ,
	cudaTextureAddressMode addressModeY ,
	cudaTextureAddressMode addressModeZ 
)
{
	// Read current volume
	this->volume_IO.readVolume(idx);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO.getField_float();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	volumeTexture.initialize(Array2Int3(solverOptions->gridSize), true, addressModeX, addressModeY, addressModeZ);
	// release host memory
	volume_IO.release();
}


void CUDASolver::loadTextureCompressed
(
	SolverOptions * solverOptions,
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ
)
{

	Timer timer;

	// Read current volume
	this->volume_IO.readVolume(idx, solverOptions);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO.getField_float_GPU();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	TIMELAPSE(volumeTexture.initialize_devicePointer(Array2Int3(solverOptions->gridSize), true, addressModeX, addressModeY, addressModeZ),"Initialize Texture including DDCopy");
	// release host memory
	//volume_IO.release();
	cudaFree(h_VelocityField);
}