#include "StreamlineSolver.h"
#include "helper_math.h"

#include "texture_fetch_functions.h"
#include "..//VolumeIO/BinaryWriter.h"
#include "ParticleTracingHelper.h"





__host__ bool StreamlineSolver::loadVolumeTexture()
{
	switch (fieldOptions->isCompressed)
	{
	case true: // Compressed
	{

		loadTextureCompressed(volumeTexture, solverOptions->currentIdx);

		break;
	}
	case false: // Uncompressed
	{
		//this->volume_IO.Initialize(solverOptions);
		this->loadTexture(volumeTexture, solverOptions->currentIdx);

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

	this->initializeParticles();
	timer.Stop();
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int blocks = BLOCK_THREAD(solverOptions->lines_count);
	ParticleTracing::TracingStream << <blocks, thread >> > (this->d_Particles, volumeTexture.getTexture(), *solverOptions,*fieldOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));
	cudaDeviceSynchronize();
	
	std::printf("time to trace particles takes %f ms \n", timer.GetMilisecondsElapsed());

	// No need for particles and volumeIO
	cudaFree(this->d_Particles);

	return true;
}

