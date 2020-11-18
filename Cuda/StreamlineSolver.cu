#include "StreamlineSolver.h"
#include "helper_math.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "texture_fetch_functions.h"
#include "..//VolumeIO/BinaryWriter.h"







__host__ bool StreamlineSolver::loadVolumeTexture()
{
	switch (solverOptions->Compressed)
	{
	case true: // Compressed
	{


		// Initialize Volume IO (Save file path and file names)
		this->volume_IO.InitializeBufferRealTime(this->solverOptions);
		loadTextureCompressed(solverOptions, volumeTexture, solverOptions->currentIdx);

		break;
	}
	case false: // Uncompressed
	{

		this->volume_IO.Initialize(solverOptions);
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
			
	this->InitializeParticles(static_cast<SeedingPattern>( this->solverOptions->seedingPattern));

	int blockDim = 1024;
	int thread = (this->solverOptions->lines_count / blockDim)+1;

	
	TracingStream << <blockDim , thread >> > (this->d_Particles, volumeTexture.getTexture(), *this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	// No need for particles and volumeIO
	cudaFree(this->d_Particles);

	return true;
}



__host__ bool StreamlineSolver::InitializeVorticityTexture()
{

	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));


	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->a_Measure.getArray();


	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;


	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_measure, &resDesc, &texDesc, NULL));



	return true;

}