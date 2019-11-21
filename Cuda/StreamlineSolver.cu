#include "StreamlineSolver.h"
#include "helper_math.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "texture_fetch_functions.h"
#include "..//Volume/BinaryWriter.h"




__host__ void StreamlineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);
	this->volumeTexture.release();
}

__host__ bool StreamlineSolver::solve()
{
	// Read Dataset
	this->volume_IO.Initialize(this->solverOptions);
	this->volume_IO.readVolume(this->solverOptions->currentIdx);

	this->h_VelocityField = this->volume_IO.flushBuffer_float();
	
	// Copy data to the texture memory
	this->volumeTexture.setField(h_VelocityField);
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.initialize();


	// Release it from Host
	volume_IO.release();
	

	this->InitializeParticles(static_cast<SeedingPattern>( this->solverOptions->seedingPattern));

	int blockDim = 1024;
	int thread = (this->solverOptions->lines_count / blockDim)+1;

	
	TracingStream << <blockDim , thread >> > (this->d_Particles, volumeTexture.getTexture(), *this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	this->release();

	return true;
}


__host__ bool StreamlineSolver::solveAndWrite()
{
	// Read Dataset
	//this->volume_IO.Initialize(this->solverOptions);
	//this->volume_IO.readVolume(this->solverOptions->currentIdx);

	//this->h_VelocityField = this->volume_IO.flushBuffer_float();

	// Copy data to the texture memory
	//this->volumeTexture.setField(h_VelocityField);
	//this->volumeTexture.setSolverOptions(this->solverOptions);
	//this->volumeTexture.initialize();

	this->measureFieldGeneration();
	// Release it from Host
	//volume_IO.release();

	float4* d_vertexBuffer;
	float4* h_vertexBuffer = new float4[solverOptions->lines_count * solverOptions->lineLength];
	cudaMalloc(&d_vertexBuffer, solverOptions->lines_count * solverOptions->lineLength * sizeof(float4));


	this->InitializeParticles(static_cast<SeedingPattern>(this->solverOptions->seedingPattern));

	int blockDim = 1024;
	int thread = (this->solverOptions->lines_count / blockDim) + 1;

	InitializeVorticityTexture();
	
	TracingStream << <blockDim, thread >> > (this->d_Particles, volumeTexture.getTexture(), t_measure ,*this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer), d_vertexBuffer);

	this->release();

	cudaMemcpy(h_vertexBuffer, d_vertexBuffer, solverOptions->lines_count * solverOptions->lineLength * sizeof(float4), cudaMemcpyDeviceToHost);

	BinaryWriter binaryWriter;
	std::string fileName = std::string(this->solverOptions->fileName_out);
	fileName += "_";
	fileName += std::to_string(solverOptions->counter);

	binaryWriter.setFileName(fileName);
	binaryWriter.setFilePath(this->solverOptions->filePath_out);
	binaryWriter.setBufferSize(solverOptions->lines_count * solverOptions->lineLength * sizeof(float4));
	binaryWriter.setBuffer(reinterpret_cast<char*>(h_vertexBuffer));
	binaryWriter.write();
	
	cudaFree(d_vertexBuffer);
	delete[] h_vertexBuffer;
	cudaDestroyTextureObject(t_measure);
	this->a_Measure.release();
	return true;
}


__host__ void StreamlineSolver::measureFieldGeneration()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { 16,16,1 };
	int gridpoints = solverOptions->gridSize[0];
	blocks = static_cast<unsigned int>((gridpoints % (thread.x * thread.y) == 0 ?
		gridpoints / (thread.x * thread.y) : gridpoints / (thread.x * thread.y) + 1));

	// set and initialize the CUDA 3D array
	this->a_Measure.setDimension
	(
		solverOptions->gridSize[0],
		solverOptions->gridSize[1],
		solverOptions->gridSize[2]
	);

	this->a_Measure.initialize();

	// initialize the CUDA surface 

	cudaArray_t pCudaArray = NULL;

	pCudaArray = this->a_Measure.getArray();


	this->s_Measure.setInputArray(pCudaArray);
	this->s_Measure.initializeSurface();


	// initialize velocity volume
	// Read Dataset
	this->volume_IO.Initialize(this->solverOptions);
	this->volume_IO.readVolume(this->solverOptions->currentIdx);

	this->h_VelocityField = this->volume_IO.flushBuffer_float();

	// Copy data to the texture memory
	this->volumeTexture.setField(h_VelocityField);
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.initialize();

	
	// Release it from Host
	volume_IO.release();



	Vorticity << <blocks, thread >> >
		(
			volumeTexture.getTexture(),
			*this->solverOptions,
			s_Measure.getSurfaceObject()
			);

	cudaDestroySurfaceObject(this->s_Measure.getSurfaceObject());

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