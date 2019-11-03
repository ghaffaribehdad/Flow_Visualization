#include "fluctuationHeightfield.h"

bool FluctuationHeightfield::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

	// initialize volume Input Output
	// Assign indexex ( from first to last idx)
	volume_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray3D
	(
		solverOptions->gridSize[2], //=> spanwise(z) gridsize
		solverOptions->lastIdx - solverOptions->firstIdx,
		solverOptions->gridSize[1]/2 //=> half of wall-normal grid size
	))
		return false;

	// Bind the array of heights to the cuda surface
	if (!this->InitializeHeightSurface3D())
		return false;

	// Here the trace and gradient must be calculate

	this->heightSurface3D.destroySurface();

	if (!this->InitializeHeightTexture3D())
		return false;


	return true;
}

void  FluctuationHeightfield::traceFluctuationfield()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	for (int i = solverOptions->firstIdx; i <= solverOptions->lastIdx; i++)
	{
		// loads the velocityfield into the memoery (accessable by field pointer)
		this->LoadVelocityfield(i);

		// 
		this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
		this->volume_IO.release();


	}

}