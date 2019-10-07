#include "DispersionTracer.h"
#include "DispersionHelper.h"
#include "..//ErrorLogger/ErrorLogger.h"



bool DispersionTracer::initialize()
{
	// Seed and initialize particles in a linear array
	if (!this->InitializeParticles())
		return false;

	// Read and store velocity field
	if (!this->InitializeVelocityField(this->solverOptions->currentIdx))
		return false;

	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray())
		return false;

	return true;
}

void DispersionTracer::setResources(SolverOptions* _solverOption, DispersionOptions* _dispersionOptions)
{
	this->solverOptions			= _solverOption;
	this->dispersionOptions		= _dispersionOptions;
}


__host__ bool DispersionTracer::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1]];
	seedParticle_ZY_Plane(h_particle, solverOptions->gridDiameter, dispersionOptions->gridSize_2D, dispersionOptions->seedWallNormalDist);


	size_t Particles_byte = sizeof(Particle) * n_particles;

	gpuErrchk(cudaMalloc((void**)& this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] this->h_particle;

	return true;
}

__host__ bool DispersionTracer::InitializeHeightArray()
{	
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->heightArray.setDimension
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1],
		dispersionOptions->timeStep
	);

	// initialize the 3D array
	if (!heightArray.initialize())
		return false;

	return true;
}

__host__ bool DispersionTracer::InitializeHeightSurface()
{
	cudaArray_t pCudaArray = NULL;
	pCudaArray = heightArray.getArray();
	this->heightSurface.setInputArray(pCudaArray);
	if (!this->heightSurface.initializeSurface())
		return false;

	return true;
}


__host__ bool DispersionTracer::InitializeVelocityField(int ID)
{
	if (!this->volume_IO.readVolume(ID))
		return false;

	std::vector<char>* p_vec_buffer = volume_IO.flushBuffer();
	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));
	this->field = reinterpret_cast<float*>(p_vec_buffer_temp);

	// pass a pointer to the velocity field to the volume texture
	this->volumeTexture.setField(this->field);
	this->volumeTexture.setSolverOptions(this->solverOptions);
	// initialize the volume texture
	if (!this->volumeTexture.initialize())
		return false;

	return true;

}

// Release resources 
void DispersionTracer::release()
{
	// Host side
	this->volume_IO.release();

	// Device Side
	this->volumeTexture.release();
	this->heightArray.release();
	this->heightSurface.destroySurface();
}

void DispersionTracer::trace()
{

}