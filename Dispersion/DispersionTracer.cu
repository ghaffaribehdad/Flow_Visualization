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
	this->rays = (*this->width) * (*this->height);

	if (!this->InitializeParticles())
		return false;


	// Depending on the Rendering mode initialize single or double surface
	if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE)
	{
		singleSurfaceInitialization();
	}
	else
	{
		doubleSurfaceInitialization();
	}



	return true;
}

void HeightfieldGenerator::setResources(Camera* _camera,
	int* _width,
	int* _height,
	SolverOptions* _solverOption,
	RaycastingOptions* _raycastingOptions,
	RenderingOptions* _renderingOptions,
	ID3D11Device* _device,
	IDXGIAdapter* _pAdapter,
	ID3D11DeviceContext* _deviceContext,
	DispersionOptions* _dispersionOptions)
{
	Raycasting::setResources(_camera, _width,_height,_solverOption,_raycastingOptions,_renderingOptions,_device,_pAdapter,_deviceContext);
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



__host__ bool HeightfieldGenerator::InitializeHeightArray3D_Single(int x, int y, int z)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->a_HeightSurface_Primary.setDimension(x, y, z);


	this->a_HeightSurface_Primary_Ex.setDimension(x, y, z);

	// initialize the 3D array
	if (!a_HeightSurface_Primary.initialize())
		return false;
	if (!a_HeightSurface_Primary_Ex.initialize())
		return false;

	return true;
}




__host__ bool HeightfieldGenerator::InitializeHeightArray3D_Single(int3 gridSize)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->a_HeightSurface_Primary.setDimension(gridSize.x, gridSize.y, gridSize.z);


	this->a_HeightSurface_Primary_Ex.setDimension(gridSize.x, gridSize.y, gridSize.z);

	// initialize the 3D array
	if (!a_HeightSurface_Primary.initialize())
		return false;
	if (!a_HeightSurface_Primary_Ex.initialize())
		return false;

	return true;
}



__host__ bool HeightfieldGenerator::InitializeHeightArray3D_Double(int x, int y, int z)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->a_HeightSurface_Primary.setDimension(x, y, z);
	this->a_HeightSurface_Secondary.setDimension(x, y, z);


	this->a_HeightSurface_Primary_Ex.setDimension(x, y, z);
	this->a_HeightSurface_Secondary_Ex.setDimension(x, y, z);

	// initialize the 3D array
	if (!a_HeightSurface_Primary.initialize())
		return false;
	if (!a_HeightSurface_Primary_Ex.initialize())
		return false;


	if (!a_HeightSurface_Secondary.initialize())
		return false;
	if (!a_HeightSurface_Secondary_Ex.initialize())
		return false;

	return true;
}




__host__ bool HeightfieldGenerator::InitializeHeightArray3D_Double(int3 gridSize)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->a_HeightSurface_Primary.setDimension(gridSize.x, gridSize.y, gridSize.z);
	this->a_HeightSurface_Secondary.setDimension(gridSize.x, gridSize.y, gridSize.z);


	this->a_HeightSurface_Primary_Ex.setDimension(gridSize.x, gridSize.y, gridSize.z);
	this->a_HeightSurface_Secondary_Ex.setDimension(gridSize.x, gridSize.y, gridSize.z);

	// initialize the 3D array
	if (!a_HeightSurface_Primary.initialize())
		return false;
	if (!a_HeightSurface_Primary_Ex.initialize())
		return false;

	if (!a_HeightSurface_Secondary.initialize())
		return false;
	if (!a_HeightSurface_Secondary_Ex.initialize())
		return false;

	return true;
}



__host__ bool HeightfieldGenerator::InitializeHeightSurface3D_Single()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;
	cudaArray_t pCudaArray_extra = NULL;

	pCudaArray = a_HeightSurface_Primary.getArray();
	pCudaArray_extra = a_HeightSurface_Primary_Ex.getArray();

	this->s_HeightSurface_Primary.setInputArray(pCudaArray);
	if (!this->s_HeightSurface_Primary.initializeSurface())
		return false;

	this->s_HeightSurface_Primary_Ex.setInputArray(pCudaArray_extra);
	if (!this->s_HeightSurface_Primary_Ex.initializeSurface())
		return false;

	return true;
}


__host__ bool HeightfieldGenerator::InitializeHeightSurface3D_Double()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;
	cudaArray_t pCudaArray_extra = NULL;

	pCudaArray = a_HeightSurface_Primary.getArray();
	pCudaArray_extra = a_HeightSurface_Primary_Ex.getArray();


	this->s_HeightSurface_Primary.setInputArray(pCudaArray);
	this->s_HeightSurface_Primary_Ex.setInputArray(pCudaArray_extra);

	
	if (!this->s_HeightSurface_Primary.initializeSurface())
		return false;

	if (!this->s_HeightSurface_Primary_Ex.initializeSurface())
		return false;


	//##########	Secondary #################

	pCudaArray = a_HeightSurface_Secondary.getArray();
	pCudaArray_extra = a_HeightSurface_Secondary_Ex.getArray();

	this->s_HeightSurface_Secondary.setInputArray(pCudaArray);
	this->s_HeightSurface_Secondary_Ex.setInputArray(pCudaArray_extra);


	if (!this->s_HeightSurface_Secondary.initializeSurface())
		return false;

	if (!this->s_HeightSurface_Secondary_Ex.initializeSurface())
		return false;

	return true;
}


// Release resources 
bool HeightfieldGenerator::release()
{
	Raycasting::release();
	cudaDestroyTextureObject(this->t_HeightSurface_Primary);
	this->a_HeightSurface_Primary.release();

	return true;
}

void HeightfieldGenerator::trace3D_path_Single()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	RK4STEP RK4Step = RK4STEP::ODD;
	
	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx ; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			this->LoadVelocityfield(i + solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
			// Release the velocityfield from host (volume_IO)
			primary_IO.release();

			// Same procedure for the second texture
			this->LoadVelocityfield(i+ solverOptions->currentIdx + 1);
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
			primary_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{
				
				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
				primary_IO.release();

				RK4Step = RK4STEP::ODD;
			}
			// Odd integration steps
			else
			{
				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
				primary_IO.release();

				RK4Step = RK4STEP::EVEN;

			}

		}

		// initialize proper velocityfield

		// trace
		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Primary.getSurfaceObject(),
				s_HeightSurface_Primary_Ex.getSurfaceObject(),
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



void HeightfieldGenerator::trace3D_path_Double()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	RK4STEP RK4Step = RK4STEP::ODD;

	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			this->LoadVelocityfield(i + solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
			// Release the velocityfield from host (volume_IO)
			primary_IO.release();

			// Same procedure for the second texture
			this->LoadVelocityfield(i + solverOptions->currentIdx + 1);
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
			primary_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{

				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
				primary_IO.release();

				RK4Step = RK4STEP::ODD;
			}
			// Odd integration steps
			else
			{
				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
				primary_IO.release();

				RK4Step = RK4STEP::EVEN;

			}

		}

		// initialize proper velocityfield

		// trace
		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Primary.getSurfaceObject(),
				s_HeightSurface_Primary_Ex.getSurfaceObject(),
				this->velocityField_0.getTexture(),
				this->velocityField_1.getTexture(),
				*solverOptions,
				*dispersionOptions,
				RK4Step,
				i
				);
	}



	/*###############################################
	#												#
	#				For the second field			#
	#												#
	################################################*/

	cudaFree(d_particle);
	this->InitializeParticles();

	RK4Step = RK4STEP::ODD;

	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			secondary_IO.readVolume(i + solverOptions->currentIdx);
			this->field = secondary_IO.flushBuffer_float();


			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
			// Release the velocityfield from host (volume_IO)
			secondary_IO.release();

			// Same procedure for the second texture
			secondary_IO.readVolume(i + solverOptions->currentIdx+1);
			this->field = secondary_IO.flushBuffer_float();


			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
			secondary_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{

				secondary_IO.readVolume(i + solverOptions->currentIdx);
				this->field = secondary_IO.flushBuffer_float();


				this->velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
				secondary_IO.release();

				RK4Step = RK4STEP::ODD;
			}
			// Odd integration steps
			else
			{

				secondary_IO.readVolume(i + solverOptions->currentIdx);
				this->field = secondary_IO.flushBuffer_float();

				this->velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
				secondary_IO.release();

				RK4Step = RK4STEP::EVEN;

			}

		}

		// initialize proper velocityfield

		// trace
		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Secondary.getSurfaceObject(),
				s_HeightSurface_Secondary_Ex.getSurfaceObject(),
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
			s_HeightSurface_Primary.getSurfaceObject(),
			s_HeightSurface_Primary_Ex.getSurfaceObject(),
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
	//this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));
 
	

	// Depending on the Rendering mode choose the terrain Rendering function
	if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE)
	{
		CudaTerrainRenderer_extra<IsosurfaceHelper::Position> << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->t_HeightSurface_Primary,
				this->t_HeightSurface_Primary_Ex,
				int(this->rays),
				this->raycastingOptions->samplingRate_0,
				this->raycastingOptions->tolerance_0,
				*dispersionOptions,
				solverOptions->lastIdx - solverOptions->firstIdx
				);
	}
	else
	{
		CudaTerrainRenderer_extra_double<IsosurfaceHelper::Position> << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->t_HeightSurface_Primary,
				this->t_HeightSurface_Primary_Ex,
				this->t_HeightSurface_Secondary,
				this->t_HeightSurface_Secondary_Ex,
				int(this->rays),
				this->raycastingOptions->samplingRate_0,
				this->raycastingOptions->tolerance_0,
				*dispersionOptions,
				solverOptions->lastIdx - solverOptions->firstIdx
				);
	}



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





bool HeightfieldGenerator::InitializeHeightTexture3D_Single()
{


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc viewDes;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&viewDes, 0, sizeof(viewDes));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->a_HeightSurface_Primary.getArray();

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Primary, &resDesc, &texDesc, NULL));


	// Use same properties with another array
	resDesc.res.array.array = this->a_HeightSurface_Primary_Ex.getArray();
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Primary_Ex, &resDesc, &texDesc, NULL));

	return true;
}



bool HeightfieldGenerator::InitializeHeightTexture3D_Double()
{


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc viewDes;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&viewDes, 0, sizeof(viewDes));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->a_HeightSurface_Primary.getArray();


	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Primary, &resDesc, &texDesc, NULL));


	// Use same properties with another array
	resDesc.res.array.array = this->a_HeightSurface_Primary_Ex.getArray();
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Primary_Ex, &resDesc, &texDesc, NULL));


	//#######################	Secondary	######################################

	resDesc.res.array.array = this->a_HeightSurface_Secondary.getArray();
	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Secondary, &resDesc, &texDesc, NULL));

	resDesc.res.array.array = this->a_HeightSurface_Secondary_Ex.getArray();
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Secondary_Ex, &resDesc, &texDesc, NULL));



	return true;
}






void HeightfieldGenerator::gradient3D_Single()
{

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	heightFieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			s_HeightSurface_Primary.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
		);


}


void HeightfieldGenerator::gradient3D_Double()
{

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	heightFieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			s_HeightSurface_Primary.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
			);


	heightFieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			s_HeightSurface_Secondary.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
			);


}



bool HeightfieldGenerator::LoadVelocityfield(const unsigned int& idx)
{

	if (!primary_IO.readVolume(idx))
		return false;

	this->field = primary_IO.flushBuffer_float();

	return true;
}



bool HeightfieldGenerator::singleSurfaceInitialization()
{
	// initialize volume Input Output
	primary_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty cuda array 3D
	if (!this->InitializeHeightArray3D_Single
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1],
		solverOptions->lastIdx - solverOptions->firstIdx
	))
		return false;



	// Bind the array of heights to the cuda surface
	if (!this->InitializeHeightSurface3D_Single())
		return false;


	// Trace particle and store their heights on the Height Surface
	this->trace3D_path_Single();


	// Store gradient and height on the surface
	this->gradient3D_Single();


	this->s_HeightSurface_Primary.destroySurface();

	if (!this->InitializeHeightTexture3D_Single())
		return false;

	return true;
}


bool HeightfieldGenerator::doubleSurfaceInitialization()
{

	// initialize Primary Volume Input Output Object
	primary_IO.Initialize(this->solverOptions);

	// initialize Secondary Volume Input Output Object
	secondary_IO.Initialize(this->solverOptions);
	secondary_IO.setFileName(this->dispersionOptions->fileNameSecondary);
	secondary_IO.setFilePath(this->dispersionOptions->filePathSecondary);

	// Initialize Height Field as an empty CUDA array 3D
	if (!this->InitializeHeightArray3D_Double
	(
		dispersionOptions->gridSize_2D[0],
		dispersionOptions->gridSize_2D[1],
		solverOptions->lastIdx - solverOptions->firstIdx
	))
		return false;



	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D_Double())
		return false;


	// Trace particle and store their heights on the Height Surface
	this->trace3D_path_Double();


	// Store gradient and height on the surface
	this->gradient3D_Double();


	this->s_HeightSurface_Primary.destroySurface();
	this->s_HeightSurface_Secondary.destroySurface();

	if (!this->InitializeHeightTexture3D_Double())
		return false;

	return true;
}



bool HeightfieldGenerator::initializeShaders()
{

	if (this->vertexBuffer.Get() == nullptr)
	{
		std::wstring shaderfolder;
#pragma region DetermineShaderPath
		if (IsDebuggerPresent() == TRUE)
		{
#ifdef _DEBUG //Debug Mode
#ifdef _WIN64 //x64
			shaderfolder = L"x64\\Debug\\";
#else //x86
			shaderfolder = L"Debug\\"
#endif // DEBUG
#else //Release mode
#ifdef _WIN64 //x64
			shaderfolder = L"x64\\Release\\";
#else  //x86
			shaderfolder = L"Release\\"
#endif // Release
#endif // _DEBUG or Release mode
		}

		D3D11_INPUT_ELEMENT_DESC layout[] =
		{
			{
				"POSITION",
				0,
				DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,
				0,
				D3D11_APPEND_ALIGNED_ELEMENT,
				D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,
				0
			},

			{
				"TEXCOORD",
				0,
				DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT,
				0,
				D3D11_APPEND_ALIGNED_ELEMENT,
				D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,
				0
			}
		};

		UINT numElements = ARRAYSIZE(layout);

		if (!vertexshader.Initialize(this->device, shaderfolder + L"vertexshaderTexture.cso", layout, numElements))
			return false;

		// Depending on the Rendering mode initialize single or double surface
		if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE)
		{
			if (!pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderTextureSampler.cso"))
				return false;
		}
		else
		{
			if (!pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderTextureSampler_Double.cso"))
				return false;
		}
		
	}


	return true;
}