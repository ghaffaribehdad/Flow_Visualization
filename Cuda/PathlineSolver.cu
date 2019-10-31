#include "Pathlinesolver.h"
#include "CudaHelperFunctions.h"



void PathlineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);


	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
}



__host__ bool PathlineSolver::solve()
{
	//At least two timesteps is needed
	int timeSteps = solverOptions->lastIdx - solverOptions->currentIdx;

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.Initialize(this->solverOptions);

	// Initialize Particles and upload it to GPU
	this->InitializeParticles(SeedingPattern::SEED_RANDOM);

	int blockDim = 256;
	int thread = (this->solverOptions->lines_count / blockDim) + 1;
	
	solverOptions->lineLength = timeSteps;
	bool odd = true;

	// set solverOptions once
	this->volumeTexture_0.setSolverOptions(this->solverOptions);
	this->volumeTexture_1.setSolverOptions(this->solverOptions);


	// we go through each time step and solve RK4 for even time steps the first texture is updated,
	// while the second texture is updated for odd time steps
	for (int step = 0; step < timeSteps; step++)
	{
		if (step == 0)
		{

			// For the first timestep we need to load two fields
			this->h_VelocityField = this->InitializeVelocityField(solverOptions->currentIdx);
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
			volume_IO.release();




			this->h_VelocityField = this->InitializeVelocityField(solverOptions->currentIdx+1);
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();
			volume_IO.release();

			odd = false;

		}
		// for the even timesteps we need to reload only one field (tn+1 in the first texture)
		else if (step %2 == 0)
		{
			this->h_VelocityField = this->InitializeVelocityField(solverOptions->firstIdx + 1);

			this->volumeTexture_0.release();
			this->h_VelocityField = this->InitializeVelocityField(solverOptions->firstIdx + 1);
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize();

			volume_IO.release();

			odd = true;
			
		}

		// for the odd timesteps we need to reload only one field (tn+1 in the second texture)
		else if (step % 2 != 0)
		{

			this->h_VelocityField = this->InitializeVelocityField(solverOptions->firstIdx + 1);

			this->volumeTexture_1.release();
			this->h_VelocityField = this->InitializeVelocityField(solverOptions->firstIdx + 1);
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();

			volume_IO.release();

			odd = false;
	
		}

		TracingPath << <blockDim, thread >> > 
			(
				this->d_Particles,
				volumeTexture_0.getTexture(),
				volumeTexture_1.getTexture(),
				*solverOptions,
				reinterpret_cast<Vertex*>(this->p_VertexBuffer),
				odd,
				step
			);



	}  	
	this->release();
	return true;
}



