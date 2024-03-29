#include "Heightfield.h"
#include "DispersionHelper.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include <cuda_runtime.h>
#include "..//Raycaster/Raycasting_Helper.h"
#include "..//Options/DispresionOptions.h"
#include "DispersionHelper.h"
#include "../Particle/ParticleHelperFunctions.h"



bool Heightfield::retrace()
{
	this->a_HeightArray3D.release();
	this->a_HeightArray3D_Copy.release();

	cudaDestroyTextureObject(this->volumeTexture3D_height.getTexture());
	cudaDestroyTextureObject(this->volumeTexture3D_height_extra.getTexture());

	cudaFree(d_particle);

	if (!this->InitializeParticles())
		return false;

	// Initialize Height Field as an empty cuda array 3D
	//if (!this->singleSurfaceInitialization())
	//	return false;

	return true;
}

bool Heightfield::initialize
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


	//singleSurfaceInitialization();
	
	return true;
}

void Heightfield::setResources(Camera* _camera,
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


__host__ bool Heightfield::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[n_particles];
	seedParticle_tiltedPlane
	(
		h_particle,
		Array2Float3(solverOptions->gridDiameter),
		ARRAYTOINT2(dispersionOptions->gridSize_2D),
		dispersionOptions->seedWallNormalDist,
		dispersionOptions->tilt_deg
	);

	size_t Particles_byte = sizeof(Particle) * n_particles;

	gpuErrchk(cudaMalloc((void**)& this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_particle;

	return true;
}





__host__ bool Heightfield::InitializeHeightSurface3D()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	this->s_HeightSurface.setInputArray(a_HeightArray3D.getArrayRef());
	if (!this->s_HeightSurface.initializeSurface())
		return false;

	this->s_HeightSurface_Copy.setInputArray(a_HeightArray3D_Copy.getArrayRef());
	if (!this->s_HeightSurface_Copy.initializeSurface())
		return false;

	return true;
}




// Release resources 
bool Heightfield::release()
{
	Raycasting::release();
	cudaDestroyTextureObject(this->volumeTexture3D_height.getTexture());
	this->a_HeightArray3D.release();
	this->volumeTexture3D_height.release();

	return true;
}

// Release resources 
bool Heightfield::releaseRaycasting()
{
	Raycasting::release();
	return true;
}






void Heightfield::trace3D_path_Single()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(n_particles);

	RK4STEP RK4Step = RK4STEP::EVEN;
	
	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx ; i++)
	{
		if (i == 0) // initial time step
		{
			
			// Load i 'dx field in volume_IO into field
			volume_IO.readVolume(i + solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			t_velocityField_0.setField(volume_IO.getField_float());
			t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
			// Release the velocityfield from host (volume_IO)
			volume_IO.release();



			// Same procedure for the second texture
			volume_IO.readVolume(i + solverOptions->currentIdx + 1);
			
			t_velocityField_1.setField(volume_IO.getField_float());
			t_velocityField_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
			// Release the velocityfield from host (volume_IO)
			volume_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{
				
				volume_IO.readVolume(solverOptions->currentIdx + i + 1);
				t_velocityField_1.release();
				t_velocityField_1.setField(volume_IO.getField_float());
				t_velocityField_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				volume_IO.release();

				RK4Step = RK4STEP::EVEN;
			}
			else
			{
				volume_IO.readVolume(solverOptions->currentIdx + i +1);
				t_velocityField_0.release();
				t_velocityField_0.setField(volume_IO.getField_float());
				t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				volume_IO.release();

				RK4Step = RK4STEP::ODD;

			}

		}

		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface.getSurfaceObject(),
				s_HeightSurface_Copy.getSurfaceObject(),
				this->t_velocityField_0.getTexture(),
				this->t_velocityField_1.getTexture(),
				*solverOptions,
				*dispersionOptions,
				RK4Step,
				i
			);
	}



	// Calculates the gradients and store it in the cuda surface
	cudaFree(d_particle);
}








__host__ void Heightfield::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	//this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));
 

	// Depending on the Rendering mode choose the terrain Rendering function
	CudaTerrainRenderer_Marching_extra<< < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->volumeTexture3D_height.getTexture(),
			int(this->rays),
			this->raycastingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0,
			*dispersionOptions,
			*renderingOptions,
			solverOptions->lastIdx - solverOptions->firstIdx
			);

}


bool Heightfield::updateScene()
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



//void Heightfield::gradient3D_Single()
//{
//
//	// Calculates the block and grid sizes
//	unsigned int blocks;
//	dim3 thread = { maxBlockDim,maxBlockDim,1 };
//	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
//		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));
//
//	heightFieldGradient3D<FetchTextureSurface::Channel_X> << < blocks, thread >> >
//		(
//			s_HeightSurface.getSurfaceObject(),
//			*dispersionOptions,
//			*solverOptions
//		);
//
//
//}


//bool Heightfield::singleSurfaceInitialization()
//{
//	// initialize volume Input Output
//	volume_IO.Initialize(this->fieldOptions);
//
//
//
//	// initialize the 3D array
//	if (!a_HeightArray3D.initialize(dispersionOptions->gridSize_2D[0],
//		dispersionOptions->gridSize_2D[1],
//		solverOptions->lastIdx - solverOptions->firstIdx)
//		)
//			return false;
//	if (!a_HeightArray3D_Copy.initialize(
//		dispersionOptions->gridSize_2D[0],
//		dispersionOptions->gridSize_2D[1],
//		solverOptions->lastIdx - solverOptions->firstIdx)
//		)
//		return false;
//
//
//
//
//	// Bind the array of heights to the cuda surface
//	if (!this->InitializeHeightSurface3D())
//		return false;
//
//
//	// Trace particle and store their heights on the Height Surface
//	this->trace3D_path_Single();
//
//
//	// Store gradient and height on the surface
//	this->gradient3D_Single();
//
//
//	this->s_HeightSurface.destroySurface();
//	this->s_HeightSurface_Copy.destroySurface();
//
//	volumeTexture3D_height.setArray(a_HeightArray3D.getArrayRef());
//	volumeTexture3D_height.initialize_array(false,cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp);
//
//	volumeTexture3D_height_extra.setArray(a_HeightArray3D_Copy.getArrayRef());
//	volumeTexture3D_height_extra.initialize_array(false, cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp);
//
//	return true;
//}



bool Heightfield::initializeShaders()
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

		if (!pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderTextureSampler.cso"))
			return false;
	}


	return true;
}