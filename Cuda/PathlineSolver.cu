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
			this->volumeTexture_0.initialize(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);	
			// release host memory
			volume_IO.release();
			


			// same procedure for the second field
			this->volume_IO.readVolume(solverOptions->currentIdx+1);
			this->h_VelocityField = this->volume_IO.getField_float();
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();

			volume_IO.release();

		}

		else if (step %2 == 0) // => EVEN
		{
			this->volume_IO.readVolume(solverOptions->currentIdx + step +1);
			this->h_VelocityField = this->volume_IO.getField_float();

			this->volumeTexture_1.release();
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();

			volume_IO.release();

			odd = true;
			
		}

		else if (step % 2 != 0) // => ODD
		{

			this->volume_IO.readVolume(solverOptions->currentIdx + step + 1);
			this->h_VelocityField = this->volume_IO.getField_float();

			this->volumeTexture_0.release();
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize();

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



