#include "HightfieldFTLE.h"
#include "..//Cuda/helper_math.h"
#include "..//Particle/ParticleHelperFunctions.h"
#include "DispersionHelper.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "../Raycaster/Raycasting.h"


extern __constant__  BoundingBox d_boundingBox;
extern __constant__ float3 d_raycastingColor;

__host__ bool HeightfieldFTLE::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[n_particles * FTLE_NEIGHBOR] ;
	seedParticle_ZY_Plane_FTLE
	(
		h_particle,
		ARRAYTOFLOAT3(solverOptions->gridDiameter),
		ARRAYTOINT2(dispersionOptions->gridSize_2D),
		dispersionOptions->seedWallNormalDist,
		dispersionOptions->tilt_deg,
		dispersionOptions->ftleDistance
	);

	size_t Particles_byte = sizeof(Particle) * n_particles * FTLE_NEIGHBOR;

	gpuErrchk(cudaMalloc((void**)&this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_particle;

	return true;
}



void HeightfieldFTLE::trace3D_path_Single()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(n_particles);

	RK4STEP RK4Step = RK4STEP::EVEN;

	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			this->LoadVelocityfield(i + solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, t_velocityField_0);
			// Release the velocityfield from host (volume_IO)
			primary_IO.release();

			// Same procedure for the second texture
			this->LoadVelocityfield(i + solverOptions->currentIdx + 1);
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, t_velocityField_1);
			primary_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{

				this->LoadVelocityfield(i + solverOptions->currentIdx + 1);
				this->t_velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, t_velocityField_1);
				primary_IO.release();

				RK4Step = RK4STEP::EVEN;
			}
			// Odd integration steps
			else
			{
				this->LoadVelocityfield(i + solverOptions->currentIdx + 1);
				this->t_velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, t_velocityField_0);
				primary_IO.release();

				RK4Step = RK4STEP::ODD;

			}

		}

		traceDispersion3D_path_FTLE << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Primary.getSurfaceObject(),
				s_HeightSurface_Primary_Ex.getSurfaceObject(),
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


void HeightfieldFTLE::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	//this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));



	// Depending on the Rendering mode choose the terrain Rendering function
	if (!dispersionOptions->ftleIsosurface)
	{
		if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE)
		{
			CudaTerrainRenderer_extra_FTLE << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->t_HeightSurface_Primary,
					this->t_HeightSurface_Primary_Ex,
					int(this->rays),
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0,
					*dispersionOptions,
					solverOptions->lastIdx - solverOptions->firstIdx + 1
					);
		}
	}
	else
	{
		CudaRaycasting_FTLE << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->t_HeightSurface_Primary,
				this->t_HeightSurface_Primary_Ex,
				int(this->rays),
				this->raycastingOptions->samplingRate_0,
				this->raycastingOptions->tolerance_0,
				*dispersionOptions,
				solverOptions->lastIdx - solverOptions->firstIdx + 1
				);
	}

}

bool HeightfieldFTLE::singleSurfaceInitialization()
{
	if (!dispersionOptions->ftleIsosurface)
	{
		HeightfieldGenerator::singleSurfaceInitialization();

		return true;
	}
	else
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
		this->gradient3D_Single_ftle();


		this->s_HeightSurface_Primary.destroySurface();
		this->s_HeightSurface_Primary_Ex.destroySurface();

		if (!this->InitializeHeightTexture3D_Single())
			return false;

		return true;
	}
		
}


void HeightfieldFTLE::gradient3D_Single_ftle()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	heightFieldGradient3DFTLE<< < blocks, thread >> >
		(
			s_HeightSurface_Primary_Ex.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
			);
}


__host__ bool HeightfieldFTLE::initializeBoundingBox()
{

	BoundingBox* h_boundingBox = new BoundingBox;


	h_boundingBox->gridSize = make_int3(dispersionOptions->gridSize_2D[0], dispersionOptions->gridSize_2D[1], solverOptions->lastIdx - solverOptions->firstIdx + 1);
	h_boundingBox->updateBoxFaces(ArrayFloat3ToFloat3(solverOptions->gridDiameter));
	h_boundingBox->updateAspectRatio(*width, *height);

	h_boundingBox->constructEyeCoordinates
	(
		XMFloat3ToFloat3(camera->GetPositionFloat3()),
		XMFloat3ToFloat3(camera->GetViewVector()),
		XMFloat3ToFloat3(camera->GetUpVector())
	);

	h_boundingBox->FOV = (this->FOV_deg / 360.0f) * XM_2PI;
	h_boundingBox->distImagePlane = this->distImagePlane;
	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));

	gpuErrchk(cudaMemcpyToSymbol(d_raycastingColor, this->raycastingOptions->color_0, sizeof(float3)));


	delete h_boundingBox;

	return true;
}