#include "DispersionTracer.h"
#include "DispersionHelper.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include <cuda_runtime.h>
#include "..//Raycaster/Raycasting_Helper.h"
#include "..//Options/DispresionOptions.h"

//explicit instantiation



bool HeightfieldGenerator::retrace()
{
	//this->heightArray3D.release();
	//this->heightArray3D_extra.release();

	//cudaDestroyTextureObject(this->heightFieldTexture3D);
	//cudaDestroyTextureObject(this->heightFieldTexture3D_extra);

	//if (!this->InitializeParticles())
	//	return false;

	//// Initialize Height Field as an empty cuda array 3D
	//if (!this->InitializeHeightArray3D())
	//	return false;

	//// Bind the array of heights to the cuda surface
	//if (!this->InitializeHeightSurface3D())
	//	return false;


	//// Trace particle and store their heights on the Height Surface
	//this->trace3D();


	//// Store gradient and height on the surface
	//this->gradient3D();

	////Destroy height + gradient surface and height calculations (both surface and array)
	//this->heightSurface3D.destroySurface();
	//this->heightSurface3D_extra.destroySurface();

	//// Initialize a texture and bind it to height + gradient array
	//if (!this->InitializeHeightTexture3D())
	//	return false;

	return true;
}

bool HeightfieldGenerator::initialize
(
	cudaTextureAddressMode addressMode_X ,
	cudaTextureAddressMode addressMode_Y ,
	cudaTextureAddressMode addressMode_Z 
)
{

	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels


	// initialize volume Input Output
	volume_IO.Initialize(this->solverOptions);

	if (!this->InitializeParticles())
		return false;

	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray3D
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1],
		dispersionOptions->tracingTime
	))
		return false;


	
	// Bind the array of heights to the cuda surface
	if (!this->InitializeHeightSurface3D())
		return false;


	// Trace particle and store their heights on the Height Surface
	this->trace3D_path();
	

	// Store gradient and height on the surface
	this->gradient3D();


	this->heightSurface3D.destroySurface();

	if (!this->InitializeHeightTexture3D())
		return false;

	return true;
}

void HeightfieldGenerator::setResources(Camera* _camera,
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


__host__ bool HeightfieldGenerator::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1]];
	//seedParticle_ZY_Plane(h_particle, solverOptions->gridDiameter, dispersionOptions->gridSize_2D, dispersionOptions->seedWallNormalDist);
	seedParticle_tiltedPlane(h_particle, solverOptions->gridDiameter, dispersionOptions->gridSize_2D, dispersionOptions->seedWallNormalDist, dispersionOptions->tilt_deg);

	size_t Particles_byte = sizeof(Particle) * n_particles;

	gpuErrchk(cudaMalloc((void**)& this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_particle;

	return true;
}



__host__ bool HeightfieldGenerator::InitializeHeightArray3D(int x, int y, int z)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->heightArray3D.setDimension(x, y, z);


	this->heightArray3D_extra.setDimension(x, y, z);

	// initialize the 3D array
	if (!heightArray3D.initialize())
		return false;
	if (!heightArray3D_extra.initialize())
		return false;

	return true;
}

__host__ bool HeightfieldGenerator::InitializeHeightArray3D(int3 gridSize)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->heightArray3D.setDimension(gridSize.x, gridSize.y, gridSize.z);


	this->heightArray3D_extra.setDimension(gridSize.x, gridSize.y, gridSize.z);

	// initialize the 3D array
	if (!heightArray3D.initialize())
		return false;
	if (!heightArray3D_extra.initialize())
		return false;

	return true;
}


__host__ bool HeightfieldGenerator::InitializeHeightSurface3D()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;
	cudaArray_t pCudaArray_extra = NULL;

	pCudaArray = heightArray3D.getArray();
	pCudaArray_extra = heightArray3D_extra.getArray();

	this->heightSurface3D.setInputArray(pCudaArray);
	if (!this->heightSurface3D.initializeSurface())
		return false;

	this->heightSurface3D_extra.setInputArray(pCudaArray_extra);
	if (!this->heightSurface3D_extra.initializeSurface())
		return false;

	return true;
}



// Release resources 
bool HeightfieldGenerator::release()
{
	Raycasting::release();
	cudaDestroyTextureObject(this->heightFieldTexture3D);
	this->heightArray3D.release();

	return true;
}

void HeightfieldGenerator::trace3D_path()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	RK4STEP RK4Step = RK4STEP::ODD;
	
	for (int i = 0; i < dispersionOptions->tracingTime; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			this->LoadVelocityfield(i);
			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
			// Release the velocityfield from host (volume_IO)
			volume_IO.release();

			// Same procedure for the second texture
			this->LoadVelocityfield(i+1);
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
			volume_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{
				
				this->LoadVelocityfield(i);
				this->velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
				volume_IO.release();

				RK4Step = RK4STEP::ODD;
			}
			// Odd integration steps
			else
			{
				this->LoadVelocityfield(i);
				this->velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
				volume_IO.release();

				RK4Step = RK4STEP::EVEN;

			}

		}

		// initialize proper velocityfield

		// trace
		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				heightSurface3D.getSurfaceObject(),
				heightSurface3D_extra.getSurfaceObject(),
				this->velocityField_0.getTexture(),
				this->velocityField_1.getTexture(),
				*solverOptions,
				*dispersionOptions,
				RK4Step,
				i
			);
	}



	// Calculates the gradients and store it in the cuda surface
	cudaFree(d_particle);
}

void HeightfieldGenerator::trace3D()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle
	traceDispersion3D_extra << < blocks, thread >> >
		(
			d_particle,
			heightSurface3D.getSurfaceObject(),
			heightSurface3D_extra.getSurfaceObject(),
			this->velocityField_0.getTexture(),
			*solverOptions,
			*dispersionOptions
		);


	// Calculates the gradients and store it in the cuda surface
	cudaFree(d_particle);
}


__host__ void HeightfieldGenerator::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Create a 2D texture to read hight array

	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));
 

	CudaTerrainRenderer_extra<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->heightFieldTexture3D,
			this->heightFieldTexture3D_extra,
			int(this->rays),
			this->raycastingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0, 
			*dispersionOptions
		);

}


bool HeightfieldGenerator::updateScene()
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





bool HeightfieldGenerator::InitializeHeightTexture3D()
{


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc viewDes;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&viewDes, 0, sizeof(viewDes));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->heightArray3D.getArray();

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->heightFieldTexture3D, &resDesc, &texDesc, NULL));


	// Use same properties with another array
	//resDesc.res.array.array = this->heightArray3D_extra.getArray();
	//gpuErrchk(cudaCreateTextureObject(&this->heightFieldTexture3D_extra, &resDesc, &texDesc, NULL));

	return true;
}







void HeightfieldGenerator::gradient3D()
{

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle

	heightFieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			heightSurface3D.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
		);


}

bool HeightfieldGenerator::LoadVelocityfield(const unsigned int& idx)
{

	if (!volume_IO.readVolume(idx))
		return false;

	this->field = volume_IO.flushBuffer_float();

	return true;
}