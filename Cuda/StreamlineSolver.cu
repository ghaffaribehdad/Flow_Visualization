#include "StreamlineSolver.h"
#include "helper_math.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "texture_fetch_functions.h"




__host__ void StreamlineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);
	this->volumeTexture.release();
}

__host__ bool StreamlineSolver::solve()
{
	// Read Dataset
	this->volume_IO.Initialize(this->solverOptions);
	this->volume_IO.readVolume(this->solverOptions->currentIdx);

	this->h_VelocityField = this->volume_IO.flushBuffer_float();
	
	// Copy data to the texture memory
	this->volumeTexture.setField(h_VelocityField);
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.initialize();


	// Release it from Host
	volume_IO.release();
	

	this->InitializeParticles(static_cast<SeedingPattern>( this->solverOptions->seedingPattern));

	int blockDim = 1024;
	int thread = (this->solverOptions->lines_count / blockDim)+1;
	
	TracingStream << <blockDim , thread >> > (this->d_Particles, volumeTexture.getTexture(), *this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	this->release();

	return true;
}