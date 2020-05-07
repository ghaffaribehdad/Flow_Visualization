#include "HightfieldFTLE.h"
#include "..//Cuda/helper_math.h"
#include "..//Particle/ParticleHelperFunctions.h"
#include "DispersionHelper.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "../Raycaster/Raycasting.h"
#include "../VolumeIO/BinaryWriter.h"


extern __constant__  BoundingBox d_boundingBox;
extern __constant__ float3 d_raycastingColor;

__host__ bool HeightfieldFTLE::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1];
	this->h_particle = new Particle[n_particles * FTLE_NEIGHBOR] ;
	seedParticle_ZY_Plane_FTLE
	(
		h_particle,
		Array2Float3(solverOptions->gridDiameter),
		ARRAYTOINT2(dispersionOptions->gridSize_2D),
		dispersionOptions->seedWallNormalDist,
		dispersionOptions->tilt_deg,
		dispersionOptions->initial_distance
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


	// Forward FTLE
	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			volume_IO.readVolume(solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			t_velocityField_0.setField(volume_IO.getField_float());
			t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize),false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
			// Release the velocityfield from host (volume_IO)
			volume_IO.release();


			// Same procedure for the second texture
			volume_IO.readVolume(solverOptions->currentIdx + 1);
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

				// Same procedure for the second texture
				volume_IO.readVolume(i + solverOptions->currentIdx + 1);
				t_velocityField_1.release();
				t_velocityField_1.setField(volume_IO.getField_float());
				t_velocityField_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				// Release the velocityfield from host (volume_IO)
				volume_IO.release();

				RK4Step = RK4STEP::EVEN;
			}
			// Odd integration steps
			else
			{
				// Same procedure for the second texture
				volume_IO.readVolume(i + solverOptions->currentIdx + 1);
				t_velocityField_0.release();
				t_velocityField_0.setField(volume_IO.getField_float());
				t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				// Release the velocityfield from host (volume_IO)
				volume_IO.release();

				RK4Step = RK4STEP::ODD;

			}

		}

		traceDispersion3D_path_FTLE << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Primary.getSurfaceObject(),
				s_HeightSurface_Primary_Extra.getSurfaceObject(),
				this->t_velocityField_0.getTexture(),
				this->t_velocityField_1.getTexture(),
				*solverOptions,
				*dispersionOptions,
				RK4Step,
				i
				);
	}


	//// Backward FTLE Calculations
	//for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	//{
	//	if (i == 0)
	//	{
	//		// Load i 'dx field in volume_IO into field
	//		volume_IO.readVolume(solverOptions->currentIdx);
	//		// Copy and initialize velocityfield texture
	//		t_velocityField_0.setField(volume_IO.getField_float());
	//		t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
	//		// Release the velocityfield from host (volume_IO)
	//		volume_IO.release();


	//		// Same procedure for the second texture
	//		volume_IO.readVolume(solverOptions->currentIdx - 1);
	//		t_velocityField_1.setField(volume_IO.getField_float());
	//		t_velocityField_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
	//		// Release the velocityfield from host (volume_IO)
	//		volume_IO.release();

	//	}
	//	else
	//	{
	//		// Even integration steps
	//		if (i % 2 == 0)
	//		{

	//			// Same procedure for the second texture
	//			volume_IO.readVolume(solverOptions->currentIdx - 1 - i);
	//			t_velocityField_1.release();
	//			t_velocityField_1.setField(volume_IO.getField_float());
	//			t_velocityField_1.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
	//			// Release the velocityfield from host (volume_IO)
	//			volume_IO.release();

	//			RK4Step = RK4STEP::EVEN;
	//		}
	//		// Odd integration steps
	//		else
	//		{
	//			// Same procedure for the second texture
	//			volume_IO.readVolume( solverOptions->currentIdx - 1 - i);
	//			t_velocityField_0.release();
	//			t_velocityField_0.setField(volume_IO.getField_float());
	//			t_velocityField_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
	//			// Release the velocityfield from host (volume_IO)
	//			volume_IO.release();

	//			RK4Step = RK4STEP::ODD;

	//		}

	//	}

	//	traceDispersion3D_path_FTLE << < blocks, thread >> >
	//		(
	//			d_particle,
	//			s_HeightSurface_Primary.getSurfaceObject(),
	//			s_HeightSurface_Primary_Extra.getSurfaceObject(),
	//			this->t_velocityField_0.getTexture(),
	//			this->t_velocityField_1.getTexture(),
	//			*solverOptions,
	//			*dispersionOptions,
	//			RK4Step,
	//			i,
	//			FTLE_Direction::BACKWARD_FTLE
	//			);
	//}




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
			CudaTerrainRenderer_Marching_extra_FSLE << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture3D_height.getTexture(),
					this->volumeTexture3D_height_extra.getTexture(),
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
				this->volumeTexture3D_height.getTexture(),
				this->volumeTexture3D_height_extra.getTexture(),
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
	
	// initialize volume Input Output
	volume_IO.Initialize(this->solverOptions);


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


	this->gradient3D_Single_ftle();


	this->s_HeightSurface_Primary.destroySurface();
	this->s_HeightSurface_Primary_Extra.destroySurface();
		

	this->volumeTexture3D_height.setArray(a_HeightSurface_Primary.getArrayRef());
	this->volumeTexture3D_height_extra.setArray(a_HeightSurface_Primary_Extra.getArrayRef());

	this->volumeTexture3D_height.initialize_array(false,cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp);
	this->volumeTexture3D_height_extra.initialize_array(false,cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp);

	// calculate the correlation between ftle and height
	this->correlation();
	// Store gradient and height on the surface

	return true;		
}


void HeightfieldFTLE::gradient3D_Single_ftle()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->n_particles % (thread.x * thread.y) == 0 ?
		n_particles / (thread.x * thread.y) : n_particles / (thread.x * thread.y) + 1));

	heightFieldGradient3D<FetchTextureSurface::Channel_X> << < blocks, thread >> >
		(
			s_HeightSurface_Primary.getSurfaceObject(),
			*dispersionOptions,
			*solverOptions
			);
}

void HeightfieldFTLE::correlation()
{
	//// Calculates the block and grid sizes
	//unsigned int blocks;
	//dim3 thread = { maxBlockDim,maxBlockDim,1 };
	//blocks = BLOCK_THREAD(n_particles);
	//
	//// Allocate device memory
	//gpuErrchk(cudaMalloc((void**)&d_mean_ftle, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMalloc((void**)&d_mean_height, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMalloc((void**)&d_pearson_cov, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMalloc((void**)&d_pearson_var_ftle, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMalloc((void**)&d_pearson_var_height, sizeof(float) * solverOptions->timeSteps));



	//// Initialize the mean value to zero at device
	//gpuErrchk(cudaMemset(this->d_mean_ftle, 0, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMemset(this->d_mean_height, 0, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMemset(this->d_pearson_cov, 0, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMemset(this->d_pearson_var_ftle, 0, sizeof(float) * solverOptions->timeSteps));
	//gpuErrchk(cudaMemset(this->d_pearson_var_height, 0, sizeof(float) * solverOptions->timeSteps));




	gpuErrchk(cudaMalloc((void**)&d_ftle, sizeof(float) * dispersionOptions->gridSize_2D[0]));
	gpuErrchk(cudaMalloc((void**)&d_height, sizeof(float) * dispersionOptions->gridSize_2D[0]));

	fetch_ftle_height << < 1, dispersionOptions->gridSize_2D[0] >> >
		(
			volumeTexture3D_height.getTexture(),
			volumeTexture3D_height_extra.getTexture(),
			d_height,
			d_ftle,
			*solverOptions
			);

	BinaryWriter binaryWriter;
	binaryWriter.setFileName("ftleValues.bin");
	binaryWriter.setFilePath("D:\\FTLE_HEIGHT\\");
	binaryWriter.setBufferSize(sizeof(float)*dispersionOptions->gridSize_2D[0]);

	h_ftle = new float[dispersionOptions->gridSize_2D[0]];
	h_height = new float[dispersionOptions->gridSize_2D[0]];
	gpuErrchk(cudaMemcpy(h_ftle, d_ftle, sizeof(float)*dispersionOptions->gridSize_2D[0], cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_height, d_height, sizeof(float)*dispersionOptions->gridSize_2D[0], cudaMemcpyDeviceToHost));

	binaryWriter.setBuffer(reinterpret_cast<char*>(h_ftle));
	binaryWriter.write();
	binaryWriter.setFileName("heightValues.bin");
	binaryWriter.setBuffer(reinterpret_cast<char*>(h_height));
	binaryWriter.write();


	


	//// calculate the mean values
	//textureMean << < blocks, thread >> >
	//	(
	//		volumeTexture3D_height.getTexture(),
	//		volumeTexture3D_height_extra.getTexture(),
	//		d_mean_height,
	//		d_mean_ftle,
	//		*dispersionOptions,
	//		*solverOptions
	//	);

	//pearson_terms << < blocks, thread >> >
	//	(
	//		volumeTexture3D_height.getTexture(),
	//		volumeTexture3D_height_extra.getTexture(),
	//		d_mean_height,
	//		d_mean_ftle,
	//		d_pearson_cov,
	//		d_pearson_var_ftle,
	//		d_pearson_var_height,
	//		*dispersionOptions,
	//		*solverOptions
	//		);

	//fetchftle_height << < blocks, thread >> >
	//(
	//	volumeTexture3D_height.getTexture(),
	//	volumeTexture3D_height_extra.getTexture(),
	//	d_mean_height,
	//	d_mean_ftle,
	//	*dispersionOptions,
	//	*solverOptions
	//);

	//pearson << < 1, solverOptions->timeSteps >> >
	//	(
	//		d_pearson_cov,
	//		d_pearson_var_ftle,
	//		d_pearson_var_height,
	//		*solverOptions
	//		);

	//h_pearson = new float[solverOptions->timeSteps];
	//gpuErrchk(cudaMemcpy(h_pearson, d_pearson_cov, sizeof(float)*solverOptions->timeSteps, cudaMemcpyDeviceToHost));

	//


	//gpuErrchk(cudaFree(d_mean_ftle));
	//gpuErrchk(cudaFree(d_mean_height));
	//gpuErrchk(cudaFree(d_pearson_var_ftle));
	//gpuErrchk(cudaFree(d_pearson_var_height));

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