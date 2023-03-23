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
bool CUDASolver::Initialize(SolverOptions * _solverOptions, FieldOptions * _fieldOptions)
{
	this->volumeTexture.release();
	this->solverOptions = _solverOptions;
	this->fieldOptions = _fieldOptions;
	this->InitializeCUDA();
	this->volume_IO.Initialize(_fieldOptions);

	return true;
}

bool CUDASolver::ReinitializeCUDA()
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






void CUDASolver::initializeParticles()
{

	// Create an array of particles
	Particle * h_Particles = new Particle[solverOptions->lines_count];

	switch (solverOptions->seedingPattern)
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

			seedParticleGridPoints(h_Particles, solverOptions);
			break;
		}

		case SeedingPattern::SEED_TILTED_PLANE:
		{
			float3 gridDiamter = Array2Float3(solverOptions->gridDiameter);
			seedParticle_tiltedPlane
			(
				h_Particles,
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

	gpuErrchk(cudaMemcpy(this->d_Particles, h_Particles, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_Particles;
}

void CUDASolver::loadTexture
(
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX ,
	cudaTextureAddressMode addressModeY ,
	cudaTextureAddressMode addressModeZ ,
	bool normalize
)
{
	// Read current volume
	this->volume_IO.readVolume(idx);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO.getField_float();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	volumeTexture.initialize(Array2Int3(solverOptions->gridSize), normalize, addressModeX, addressModeY, addressModeZ);
	// release host memory
	volume_IO.release();
}


void CUDASolver::loadTextureCompressed
(
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ,
	bool nomralize

)
{

	Timer timer;

	// Read current volume
	this->volume_IO.readVolume_Compressed(idx, Array2Int3(fieldOptions->gridSize));
	// Return a pointer to volume on device
	float * d_VelocityField = this->volume_IO.getField_float_GPU();
	// set the pointer to the volume texture
	volumeTexture.setField(d_VelocityField);
	// initialize the volume texture
	TIMELAPSE(volumeTexture.initialize_devicePointer(Array2Int3(fieldOptions->gridSize), nomralize, addressModeX, addressModeY, addressModeZ),"Initialize Texture including DDCopy");
	// release device memory
	cudaFree(d_VelocityField);
}