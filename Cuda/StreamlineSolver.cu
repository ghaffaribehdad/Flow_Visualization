#include "StreamlineSolver.h"
#include "helper_math.h"

#include "texture_fetch_functions.h"
#include "..//VolumeIO/BinaryWriter.h"
#include "ParticleTracingHelper.h"





__host__ bool StreamlineSolver::loadVolumeTexture()
{
	switch (solverOptions->Compressed)
	{
	case true: // Compressed
	{
		// Initialize Volume IO (Save file path and file names)
		//this->volume_IO.Initialize(this->solverOptions);
		loadTextureCompressed(solverOptions, volumeTexture, solverOptions->currentIdx);

		break;
	}
	case false: // Uncompressed
	{
		//this->volume_IO.Initialize(solverOptions);
		this->loadTexture(solverOptions, volumeTexture, solverOptions->currentIdx);

		break;
	}
	}

	return true;
}




__host__ bool StreamlineSolver::release()
{
	this->volume_IO.release();
	volumeTexture.release();

	return true;
}

__host__ bool StreamlineSolver::releaseVolumeIO()
{

	this->volume_IO.release();

	return true;
}

__host__ bool StreamlineSolver::releaseVolumeTexture()
{

	this->volumeTexture.release();


	return true;
}

__host__ bool StreamlineSolver::solve()
{
	timer.Start();

	this->initializeParticles(static_cast<SeedingPattern>(solverOptions->seedingPattern));
	timer.Stop();
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int blocks = BLOCK_THREAD(solverOptions->lines_count);
	ParticleTracing::TracingStream << <blocks, thread >> > (this->d_Particles, volumeTexture.getTexture(), *solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));
	cudaDeviceSynchronize();
	
	std::printf("time to trace particles takes %f ms \n", timer.GetMilisecondsElapsed());

	// No need for particles and volumeIO
	cudaFree(this->d_Particles);

	return true;
}

