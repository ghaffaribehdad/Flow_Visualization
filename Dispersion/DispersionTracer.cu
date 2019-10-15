#include "DispersionTracer.h"
#include "DispersionHelper.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include <cuda_runtime.h>
#include "..//Raycaster/Raycasting_Helper.h"


void DispersionTracer::retrace()
{
	this->InitializeParticles();
	trace();
	this->InitializeHeightTexture();
}
bool DispersionTracer::initialize
(
	cudaTextureAddressMode addressMode_X ,
	cudaTextureAddressMode addressMode_Y ,
	cudaTextureAddressMode addressMode_Z 
)
{

	// Seed and initialize particles in a linear array
	Raycasting::initialize(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);

	if (!this->InitializeParticles())
		return false;

	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray())
		return false;

	// Bind the array of heights to the cuda surface
	if (!this->InitializeHeightSurface())
		return false;

	if (!this->InitializeHeightTexture())
		return false;

	return true;
}

void DispersionTracer::setResources(Camera* _camera,
	int* _width,
	int* _height,
	SolverOptions* _solverOption,
	RaycastingOptions* _raycastingOptions,
	ID3D11Device* _device,
	IDXGIAdapter* _pAdapter,
	ID3D11DeviceContext* _deviceContext,
	DispersionOptions* _dispersionOptions)
{
	Raycasting::setResources(_camera, _width,_height,_solverOption,_raycastingOptions,_device,_pAdapter,_deviceContext);
		this->dispersionOptions		= _dispersionOptions;
}


__host__ bool DispersionTracer::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1]];
	seedParticle_ZY_Plane(h_particle, solverOptions->gridDiameter, dispersionOptions->gridSize_2D, dispersionOptions->seedWallNormalDist);


	size_t Particles_byte = sizeof(Particle) * n_particles;

	gpuErrchk(cudaMalloc((void**)& this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_particle;

	return true;
}

__host__ bool DispersionTracer::InitializeHeightArray()
{	
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->heightArray.setDimension
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1]
	);

	// initialize the 3D array
	if (!heightArray.initialize())
		return false;

	return true;
}

__host__ bool DispersionTracer::InitializeHeightSurface()
{
	cudaArray_t pCudaArray = NULL;
	pCudaArray = heightArray.getArray();
	this->heightSurface.setInputArray(pCudaArray);
	if (!this->heightSurface.initializeSurface())
		return false;

	// Trace particles and write their position in to the surface
	this->trace();

	return true;
}


// Release resources 
bool DispersionTracer::release()
{
	Raycasting::release();
	this->heightArray.release();
	this->heightSurface.destroySurface();
	cudaDestroyTextureObject(this->heightFieldTexture);

	return true;
}

void DispersionTracer::trace()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle
	traceDispersion << < blocks, thread >> >
	(
		d_particle,
		heightSurface.getSurfaceObject(),
		this->volumeTexture.getTexture(),
		*solverOptions,
		*dispersionOptions
	);


	cudaFree(d_particle);
}


__host__ void DispersionTracer::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	this->heightSurface.destroySurface();
	// Create a 2D texture tu read hight array

	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));
 
	int2 gridSize = { dispersionOptions->gridSize_2D[0], dispersionOptions->gridSize_2D[1] };

	CudaTerrainRenderer<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->heightFieldTexture,
			int(this->rays),
			this->raycastingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0, 
			gridSize,
			dispersionOptions->seedWallNormalDist
		);

}


bool DispersionTracer::updateScene()
{
	if (!this->initializeRaycastingInteroperability())	// Create interoperability while we need to release it at the end of rendering
		return false;

	if (!this->initializeCudaSurface())					// reinitilize cudaSurface	
		return false;

	if (!this->initializeBoundingBox())					//updates constant memory
		return false;


	this->rendering();


	if (!this->raycastingSurface.destroySurface())
		return false;

	this->interoperatibility.release();


	return true;
}

bool DispersionTracer::InitializeHeightTexture()
{
	
	this->heightSurface.destroySurface();

	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->heightArray.getArray();

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;

	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->heightFieldTexture, &resDesc, &texDesc, NULL));

	return true;
}



