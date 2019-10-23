#include "DispersionTracer.h"
#include "DispersionHelper.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include <cuda_runtime.h>
#include "..//Raycaster/Raycasting_Helper.h"
#include "..//Options/DispresionOptions.h"

//explicit instantiation



bool DispersionTracer::retrace()
{
	this->heightArray_gradient.release();
	cudaDestroyTextureObject(this->heightFieldTexture);

	if (!this->InitializeParticles())
		return false;

	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray())
		return false;

	// Bind the array of heights to the cuda surface
	if (!this->InitializeHeightSurface())
		return false;


	// Trace particle and store their heights on the Height Surface
	this->trace();

	// Initialize Array for height + gradient
	if (!this->InitializeHeight_gradient_Array())
		return false;

	// Initialize Surface to store gradient + height
	if (!this->InitializeHeight_gradient_Surface())
		return false;

	// Store gradient and height on the surface
	this->gradient();

	//Destroy height + gradient surface and height calculations (both surface and array)
	this->heightSurface.destroySurface();
	this->heightArray.release();
	this->heightSurface_gradient.destroySurface();

	// Initialize a texture and bind it to height + gradient array
	if (!this->InitializeHeight_gradient_Texture())
		return false;

	return true;
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


	// Trace particle and store their heights on the Height Surface
	this->trace();
	
	// Initialize Array for height + gradient
	if (!this->InitializeHeight_gradient_Array())
		return false;

	// Initialize Surface to store gradient + height
	if (!this->InitializeHeight_gradient_Surface())
		return false;

	// Store gradient and height on the surface
	this->gradient();

	//Destroy height + gradient surface and height calculations (both surface and array)
	this->heightSurface.destroySurface();
	this->heightArray.release();
	this->heightSurface_gradient.destroySurface();

	// Initialize a texture and bind it to height + gradient array
	if (!this->InitializeHeight_gradient_Texture())
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
	// Set dimensions and initialize height field as a 2D CUDA Array
	this->heightArray.setDimension
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1]
	);

	// initialize the 2D array
	if (!heightArray.initialize())
		return false;

	return true;
}


__host__ bool DispersionTracer::InitializeHeightArray3D()
{
	// Set dimensions and initialize height field as a 2D CUDA Array
	this->heightArray3D.setDimension
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1],
		dispersionOptions->timeStep
	);

	// initialize the 3D array
	if (!heightArray3D.initialize())
		return false;

	return true;
}


__host__ bool DispersionTracer::InitializeHeight_gradient_Array()
{
	// Set dimensions and initialize height field as a 2D CUDA Array
	this->heightArray_gradient.setDimension
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1]
	);

	// initialize the 2D array
	if (!heightArray_gradient.initialize())
		return false;

	return true;
}

__host__ bool DispersionTracer::InitializeHeightSurface()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;
	pCudaArray = heightArray.getArray();
	this->heightSurface.setInputArray(pCudaArray);
	if (!this->heightSurface.initializeSurface())
		return false;


	return true;
}

__host__ bool DispersionTracer::InitializeHeight_gradient_Surface()
{
	//  Assign the hightArray_gradient to the hightSurface_gradient and initialize the surface
	cudaArray_t pCudaArray_gradient = NULL;
	pCudaArray_gradient = heightArray_gradient.getArray();
	this->heightSurface_gradient.setInputArray(pCudaArray_gradient);
	if (!this->heightSurface_gradient.initializeSurface())
		return false;

	return true;
}


// Release resources 
bool DispersionTracer::release()
{
	Raycasting::release();
	this->heightSurface_gradient.destroySurface();
	this->heightArray_gradient.release();
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


	// Calculates the gradients and store it in the cuda surface
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
 

	CudaTerrainRenderer<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->heightFieldTexture,
			int(this->rays),
			this->raycastingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0, 
			*dispersionOptions
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



bool DispersionTracer::InitializeHeight_gradient_Texture()
{


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->heightArray_gradient.getArray();

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;

	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->heightFieldTexture, &resDesc, &texDesc, NULL));

	return true;
}




void DispersionTracer::gradient()
{

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle

	hieghtfieldGradient<IsosurfaceHelper::Position> << < blocks, thread >> >
		(

			heightSurface.getSurfaceObject(),
			heightSurface_gradient.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
		);


}

