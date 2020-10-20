#include "StreaklineSolver.h"
#include "CudaHelperFunctions.h"



void StreaklineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);


	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
}

__host__ bool StreaklineSolver::initializeRealtime()
{
	//At least two timesteps is needed
	this->timeSteps = solverOptions->lastIdx - solverOptions->firstIdx;

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.InitializeRealTime(this->solverOptions);

	// Initialize Particles and upload it to GPU
	this->InitializeParticles(this->solverOptions->seedingPattern);

	int blockDim = 256;
	int thread = (this->solverOptions->lines_count / blockDim) + 1;

	// set the position of the vertex buffer to the intial position of the particle
	InitializeVertexBufferStreaklines << <blockDim, thread >> >
		(this->d_Particles,
			*solverOptions,
			reinterpret_cast<Vertex*>(this->p_VertexBuffer)
			);

	solverOptions->lineLength = timeSteps;

	return true;
}

__host__ bool StreaklineSolver::solve()
{
	//At least two timesteps is needed
	int timeSteps = solverOptions->lastIdx - solverOptions->currentIdx;

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.Initialize(this->solverOptions);

	// Initialize Particles and upload it to GPU
	this->InitializeParticles(solverOptions->seedingPattern);

	// Number of threads based on the number of lines
	int blockDim = 256;
	int thread = (this->solverOptions->lines_count / blockDim) + 1;

	solverOptions->lineLength = timeSteps;
	bool odd = false;

	// set the position of the vertex buffer to the intial position of the particle
	InitializeVertexBufferStreaklines << <blockDim, thread >> >
		(	this->d_Particles,
			*solverOptions,
			reinterpret_cast<Vertex*>(this->p_VertexBuffer)
			);

	// we go through each time step and solve RK4 for even time steps the first texture is updated,
	// while the second texture is updated for odd time steps
	for (int step = 0; step < timeSteps; step++)
	{
		// First Step
		if (step == 0)
		{

			// Read current volume
			this->volume_IO.readVolume(solverOptions->currentIdx);
			// Return a pointer to volume
			this->h_VelocityField = this->volume_IO.getField_float();
			// set the pointer to the volume texture
			this->volumeTexture_0.setField(h_VelocityField);
			// initialize the volume texture
			this->volumeTexture_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
			// release host memory
			volume_IO.release();



			// same procedure for the second field
			this->volume_IO.readVolume(solverOptions->currentIdx + 1);
			this->h_VelocityField = this->volume_IO.getField_float();
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);

			volume_IO.release();

		}

		else if (step % 2 == 0) // => EVEN
		{
			this->volume_IO.readVolume(solverOptions->currentIdx + step + 1);
			this->h_VelocityField = this->volume_IO.getField_float();

			this->volumeTexture_1.release();
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);

			volume_IO.release();


			odd = false;
		}

		else if (step % 2 != 0) // => ODD
		{

			this->volume_IO.readVolume(solverOptions->currentIdx + step + 1);
			this->h_VelocityField = this->volume_IO.getField_float();

			this->volumeTexture_0.release();
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);

			volume_IO.release();

			odd = true;

		}

		TracingStreak << <blockDim, thread >> >
			(
				volumeTexture_0.getTexture(),
				volumeTexture_1.getTexture(),
				*solverOptions,
				reinterpret_cast<Vertex*>(this->p_VertexBuffer),
				odd,
				step
				);



	}

	// Bring the position to the middle
	AddOffsetVertexBufferStreaklines << <blockDim, thread >> >(*solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));
	this->release();
	return true;
}


__host__ bool StreaklineSolver::solveRealtime()
{
	int blockDim = 256;
	int thread = (this->solverOptions->lines_count / blockDim) + 1;

	bool odd = false;


	// First Step

	switch (solverOptions->Compressed)
	{

	case true: // Compressed Data
	{
		if (solverOptions->counter == 0)
		{
			initializeTextureCompressed(solverOptions, volumeTexture_0, solverOptions->currentIdx);
			initializeTextureCompressed(solverOptions, volumeTexture_1, solverOptions->currentIdx + 1);
		}
		else if (this->solverOptions->counter % 2 == 0) // => EVEN
		{
			this->volumeTexture_1.release();
			initializeTextureCompressed(solverOptions, volumeTexture_1, solverOptions->currentIdx + solverOptions->counter + 1);
			odd = false;
		}
		else if (this->solverOptions->counter % 2 != 0) // => ODD
		{
			this->volumeTexture_0.release();
			initializeTextureCompressed(solverOptions, volumeTexture_0, solverOptions->currentIdx + solverOptions->counter + 1);
			odd = true;

		}

		break;
	}

	case false: // Uncompressed Data
	{
		if (solverOptions->counter == 0)
		{
			initializeTexture(solverOptions, volumeTexture_0, solverOptions->currentIdx);
			initializeTexture(solverOptions, volumeTexture_1, solverOptions->currentIdx + 1);
		}
		else if (solverOptions->counter % 2 == 0) // => EVEN
		{
			this->volumeTexture_1.release();
			initializeTexture(solverOptions, volumeTexture_1, solverOptions->currentIdx + solverOptions->counter + 1);
			odd = false;

		}
		else if (solverOptions->counter % 2 != 0) // => ODD
		{
			this->volumeTexture_0.release();
			initializeTexture(solverOptions, volumeTexture_0, solverOptions->currentIdx + solverOptions->counter + 1);
			odd = true;
		}
		break;
	}

	}

	//int numofBlock = 0;
	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numofBlock, TracingPath, blockDim, 0);
	//std::printf("Optimized number of blocks are  %d \n", numofBlock);	

	TracingStreak << <blockDim, thread >> > (volumeTexture_0.getTexture(), volumeTexture_1.getTexture(), *solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer), odd, solverOptions->counter);
	std::printf("\n\n");

	if (solverOptions->counter == timeSteps)
	{
		solverOptions->drawComplete = true;
		this->release();
	}
	else
	{
		solverOptions->counter++;
	}


	return true;
}



