#include "TurbulentMixing.h"
#include "TurbulentMixingHelper.h"
#include "../Cuda/helper_math.h"

bool TurbulentMixing::initalize()
{

#pragma region Raycasting

	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

#pragma endregion

#pragma region InitializeMixingArraySurface 	
	// Create mixing array and bind it to the surface

	// Set dimensions of CUDA array
	this->a_mixing.setDimension
	(
		solverOptions->gridSize[1],
		solverOptions->gridSize[2]
	);
	
	// Initialize the CUDA Array
	if (!this->a_mixing.initialize())
		return false;

	// Assign the CUDA array to the Surface
	this->s_mixing.setInputArray(a_mixing.getArrayRef());
	
	// Initialize CUDA Surface
	if (!this->s_mixing.initializeSurface())
		return false;
#pragma endregion 

#pragma region InitializeVolumeVelocityFields

	// Read  a plane (ZY) of the first timestep and store it in v_field_t0



	volumeIO.Initialize(this->fieldOptions);

	if (!volumeIO.readVolumePlane(this->solverOptions->firstIdx, VolumeIO::readPlaneMode::YZ, turbulentMixingOptions->streamwisePlane))
		return false;


	this->v_field_t0.setField(volumeIO.getField_float());
	this->v_field_t0.initialize
	(
		make_int2(solverOptions->gridSize[0], solverOptions->gridSize[1]),
		false,
		cudaAddressModeClamp,
		cudaAddressModeClamp
	);


	volumeIO.release();


	//// Read the second one
	//if (!volumeIO.readVolumePlane(this->solverOptions->firstIdx + 1 , VolumeIO::readPlaneMode::YZ, turbulentMixingOptions->streamwisePlane, offset, buffer_size))
	//	return false;

	//this->v_field_t1.setSolverOptions(this->solverOptions);
	//this->v_field_t1.setSolverOptions(this->solverOptions);

	//this->v_field_t1.setField(volumeIO.getField_float());
	//this->v_field_t1.initialize
	//(
	//	solverOptions->gridSize[1],
	//	solverOptions->gridSize[2],
	//	cudaAddressModeClamp,
	//	cudaAddressModeClamp,
	//	cudaFilterModePoint
	//);

	//volumeIO.release();

#pragma endregion


	return true;
}

bool TurbulentMixing::updateVolume(VolumeTexture2D& v_field, int& idx)
{
	v_field.release();


	if (!volumeIO.readVolumePlane(idx, VolumeIO::readPlaneMode::YZ, turbulentMixingOptions->streamwisePlane))
		return false;


	v_field.setField(volumeIO.getField_float());
	v_field.initialize
	(
		make_int2(solverOptions->gridSize[1],solverOptions->gridSize[2]),
		false,
		cudaAddressModeClamp,
		cudaAddressModeClamp
	);

	volumeIO.release();


	return true;
}



bool TurbulentMixing::release()
{
	this->s_mixing.destroySurface();
	this->a_mixing.release();
	this->v_field_t0.release();
	//this->v_field_t1.release();

	return true;
}


void TurbulentMixing::create()
{
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	size_t meshSize = (size_t)this->solverOptions->gridSize[1] * (size_t)this->solverOptions->gridSize[2];
	blocks = BLOCK_THREAD(meshSize);

	createTKE << < blocks, thread >> >
		(
			s_mixing.getSurfaceObject(),
			v_field_t0.getTexture(),
			*solverOptions,
			*turbulentMixingOptions
			);
}

void TurbulentMixing::advect()
{
	//unsigned int blocks = 0;
	//dim3 thread = { maxBlockDim,maxBlockDim,1 };

	//size_t meshSize = this->solverOptions->gridSize[1] * this->solverOptions->gridSize[2];
	//blocks = BLOCK_THREAD(meshSize);

	//advectTKE << < blocks, thread >> >
	//	(
	//		s_mixing.getSurfaceObject(),
	//		v_field_t0.getTexture(),
	//		*solverOptions,
	//		*turbulentMixingOptions
	//		);

}

void TurbulentMixing::dissipate()
{
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	size_t meshSize = this->solverOptions->gridSize[1] * this->solverOptions->gridSize[2];
	blocks = BLOCK_THREAD(meshSize);


	dissipateTKE << < blocks, thread >> >
		(
			s_mixing.getSurfaceObject(),
			v_field_t0.getTexture(),
			*solverOptions,
			*turbulentMixingOptions
			);
}

void TurbulentMixing::update(int timestep)
{
	for (int t = 0; t < timestep; t++)
	{
		dissipate();
		create();
		advect();
	}

}

void TurbulentMixing::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());



	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view


	create();
}