#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"

bool FluctuationHeightfield::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	if (!this->initializeRaycastingTexture())				// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

	// initialize volume Input Output
	// Assign indexes ( from first to last idx)
	volume_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!this->InitializeHeightArray3D
	(
		solverOptions->gridSize[2],								//=> spanwise(z) gridsize
		solverOptions->lastIdx - solverOptions->firstIdx,		//=> # time snaps
		solverOptions->gridSize[1]								//=> wall-normal grid size
	))
		return false;

	// Bind the array of heights to the CUDA surface
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

	int nMesh = solverOptions->gridSize[2];

	blocks = static_cast<unsigned int>((nMesh % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : nMesh / (thread.x * thread.y) + 1));

	for (int i = solverOptions->firstIdx; i <= solverOptions->lastIdx; i++)
	{
		// loads the velocity field into the memory (accessible by this->field pointer)
		this->LoadVelocityfield(i);

		// Copy memory into the texture (using border mode since no particle advection desired here!)
		this->initializeVolumeTexuture(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder, velocityField_0);
		this->volume_IO.release();

		// Now is time to populate the texture
		trace_fluctuation3D << < blocks, thread >> >
			(
				heightSurface3D.getSurfaceObject(),
				this->velocityField_0.getTexture(),
				*solverOptions,
				*dispersionOptions,
				i,
				0.5f
			);
	}

}