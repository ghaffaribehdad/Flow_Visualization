#include "Pathlinesolver.h"
#include "CudaHelperFunctions.h"



bool PathlineSolver::release()
{
	this->volume_IO.release();
	cudaFree(this->d_Particles);


	this->volumeTexture_0.release();
	this->volumeTexture_1.release();

	return true;
}



__host__ bool PathlineSolver::solve()
{
	//At least two timesteps is needed
	int timeSteps = solverOptions->lineLength;

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.Initialize(this->fieldOptions);

	// Initialize Particles and upload it to GPU
	this->initializeParticles();

	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int blocks = BLOCK_THREAD(this->solverOptions->lines_count);
	
	bool odd = false;
	// we go through each time step and solve RK4. For even timesteps, the first texture is updated,
	// while the second texture is updated for odd timesteps



	for (int step = 0; step < timeSteps; step++)
	{
		// First Step

		switch (fieldOptions->isCompressed)
		{

			case true: // Compressed Data
			{
				if (step == 0)
				{
					loadTextureCompressed( volumeTexture_0, solverOptions->firstIdx);
					loadTextureCompressed( volumeTexture_1, solverOptions->firstIdx + 1);
				}
				else if (step == 1)
				{
					odd = true;
				}
				else if (step % 2 == 0) // => EVEN
				{
					this->volumeTexture_0.release();
					loadTextureCompressed( volumeTexture_0, solverOptions->firstIdx + step);
					odd = false;
				}
				else if (step % 2 != 0) // => ODD
				{
					this->volumeTexture_1.release();
					loadTextureCompressed( volumeTexture_1, solverOptions->firstIdx + step);
					odd = true;

				}

				break;
			}

			case false: // Uncompressed Data
			{
				if (step == 0)
				{
					loadTexture( volumeTexture_0, solverOptions->firstIdx);
					loadTexture( volumeTexture_1, solverOptions->firstIdx + 1);
				}
				else if (step == 1)
				{
					odd = true;
				}
				else if (step % 2 == 0) // => EVEN
				{
					this->volumeTexture_0.release();
					loadTexture( volumeTexture_0, solverOptions->firstIdx + step);
					odd = false;

				}
				else if (step % 2 != 0) // => ODD
				{
					this->volumeTexture_1.release();
					loadTexture( volumeTexture_1, solverOptions->firstIdx + step);
					odd = true;
				}
				break;
			}

		}
	

		TracingPath <<<blocks, thread >>> (this->d_Particles, volumeTexture_0.getTexture(), volumeTexture_1.getTexture(), *solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer), odd, step);
		std::printf("\n\n");
	}  	


	this->release();
	return true;
}

__host__ bool PathlineSolver::initializeRealtime(SolverOptions * p_solverOptions, FieldOptions * p_fieldOptions)
{

	this->solverOptions = p_solverOptions;
	this->InitializeCUDA();
	this->fieldOptions = p_fieldOptions;
	this->volume_IO.Initialize(fieldOptions);
	this->initializeParticles();
	
	return true;
}


__host__ bool PathlineSolver::resetRealtime()
{

	this->release();
	
	return true;
}

__host__ bool PathlineSolver::solveRealtime(int & pathCounter)
{

	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int blocks = BLOCK_THREAD(this->solverOptions->lines_count);
	bool odd = false;


	// First Step

	switch (fieldOptions->isCompressed)
	{

	case true: // Compressed Data
	{
		if (pathCounter == 0)
		{
			loadTextureCompressed( volumeTexture_0, solverOptions->firstIdx);
			loadTextureCompressed( volumeTexture_1, solverOptions->firstIdx + 1);
		}
		else if (pathCounter == 1)
		{
			odd = true;
		}
		else if (pathCounter % 2 == 0) // => EVEN
		{
			this->volumeTexture_1.release();
			loadTextureCompressed( volumeTexture_1, solverOptions->firstIdx + pathCounter + 1);
			odd = false;
		}
		else if (pathCounter % 2 != 0) // => ODD
		{
			this->volumeTexture_0.release();
			loadTextureCompressed( volumeTexture_0, solverOptions->firstIdx + pathCounter + 1);
			odd = true;

		}

		break;
	}

	case false: // Uncompressed Data
	{
		if (pathCounter == 0)
		{
			loadTexture( volumeTexture_0, solverOptions->firstIdx);
			loadTexture( volumeTexture_1, solverOptions->firstIdx + 1);
		}
		else if (pathCounter == 1)
		{
			odd = true;
		}
		else if (pathCounter % 2 == 0) // => EVEN
		{
			this->volumeTexture_1.release();
			loadTexture(volumeTexture_1, solverOptions->firstIdx + pathCounter + 1);
			odd = false;

		}
		else if (pathCounter % 2 != 0) // => ODD
		{
			this->volumeTexture_0.release();
			loadTexture(volumeTexture_0, solverOptions->firstIdx + pathCounter + 1);
			odd = true;
		}
		break;
	}

	}

	if (pathCounter == solverOptions->lineLength - 1)
	{
		resetRealtime();
		pathCounter = 0;

	}
	else
	{
		TracingPath << <blocks, thread >> > (this->d_Particles, volumeTexture_0.getTexture(), volumeTexture_1.getTexture(), *solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer), odd, pathCounter);
		std::printf("\n\n");
		pathCounter++;
	}
	
	solverOptions->counter = pathCounter;
	
	return true;
}


