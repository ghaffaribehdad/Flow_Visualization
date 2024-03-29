#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Raycaster/Raycasting_Helper.h"

extern __constant__  BoundingBox d_boundingBox;
extern __constant__  BoundingBox d_boundingBox_spacetime;




bool FluctuationHeightfield::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->m_gridSize3D =
	{
		1 + solverOptions->lastIdx - solverOptions->firstIdx,
		solverOptions->gridSize[1],
		solverOptions->gridSize[2]

	};


	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	this->rays = (*this->width) * (*this->height);				// Set number of rays based on the number of pixels

	// initialize volume Input Output
	this->volume_IO.Initialize(&this->fieldOptions[0]);


	// Initialize Height Field as an empty CUDA array 3D
	if (!a_HeightArray3D.initialize(m_gridSize3D.x, m_gridSize3D.y, m_gridSize3D.z))
		return false;


	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D(a_HeightArray3D.getArray()))
		return false;

	// Trace the fluctuation field
	//this->generateTimeSpaceField3D(spaceTimeOptions);

	// Destroy the surface
	this->s_HeightSurface.destroySurface();

	volumeTexture3D_height.setArray(a_HeightArray3D.getArrayRef());

	this->volumeTexture3D_height.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);

	return true;
}


void  FluctuationHeightfield::generateTimeSpaceField3D(SpaceTimeOptions * timeSpaceOptions)
{

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	blocks = BLOCK_THREAD(m_gridSize3D.z); // Kernel calls are based on the Spanwise gridSize

	Timer timer;
	
	for (int t = 0; t < m_gridSize3D.x; t++) // Iterates over time
	{
		
		// First Read the Compressed file and move it to GPU
		if (fieldOptions->isCompressed)
		{
			TIMELAPSE(this->volume_IO.readVolume_Compressed(t + solverOptions->firstIdx, Array2Int3(solverOptions->gridSize)), "read volume compress");
			std::printf("\n\n");
			// Copy the device pointer
			float * h_VelocityField = this->volume_IO.getField_float_GPU();

			// Bind and copy device pointer to texture
			volumeTexture_0.setField(h_VelocityField);
			TIMELAPSE(volumeTexture_0.initialize_devicePointer(Array2Int3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder), "initialize texture takes");
			cudaFree(h_VelocityField);

		}
		else
		{
			TIMELAPSE(this->volume_IO.readVolume(t + solverOptions->firstIdx), "read volume compress");
			// Copy the device pointer
			float * h_VelocityField = this->volume_IO.getField_float();

			// Bind and copy device pointer to texture
			volumeTexture_0.setField(h_VelocityField);
			TIMELAPSE(volumeTexture_0.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder), "initialize texture takes");
		}

		// Copy from texture to surface
		copyTextureToSurface<< < blocks, thread >> >
			(
				solverOptions->projectPos, //		Streamwise Pos
				t, //		Timestep
				*solverOptions,
				*spaceTimeOptions,
				volumeTexture_0.getTexture(),
				s_HeightSurface.getSurfaceObject()
				);

		// Release the volume texture
		volumeTexture_0.release();
		
	}

	volume_IO.release();
	solverOptions->compressResourceInitialized = false;
}



void FluctuationHeightfield::gaussianFilter()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	blocks = BLOCK_THREAD((int)m_gridSize3D.y);

	//applyGaussianFilter << < blocks, thread >> > 
	//(
	//	9,
	//	{ m_gridSize3D.x,m_gridSize3D.y,m_gridSize3D.z},
	//	this->volumeTexture3D_height.getTexture(),
	//	this->s_HeightSurface.getSurfaceObject()
	//);


}

__host__ void FluctuationHeightfield::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Odd filter Size
	if (spaceTimeOptions->filterSize % 2 == 0)
		spaceTimeOptions->filterSize++;

	// Create a 2D texture to read hight array

	int current = solverOptions->currentIdx;
	float timeDim = solverOptions->timeDim;
	int lastIdx = solverOptions->lastIdx;
	int	firstIdx = solverOptions->firstIdx;

	float init_pos = -timeDim / 2;
	init_pos += (current - firstIdx) * (timeDim / (lastIdx - firstIdx));


	this->spaceTimeOptions->projectionPlanePos = init_pos;
	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));

	spaceTimeOptions->currentTime = solverOptions->currentIdx - solverOptions->firstIdx ;

	if (spaceTimeOptions->additionalRaycasting)
	{
		if (!spaceTimeOptions->volumeLoaded)
		{
			loadRaycastingTexture(&fieldOptions[0], solverOptions->currentIdx);
			spaceTimeOptions->volumeLoaded = true;
			volume_IO.release();
		}
	}


	CudaTerrainRenderer_height_isoProjection << < blocks, thread >> >
	(
		this->raycastingSurface.getSurfaceObject(),
		this->volumeTexture3D_height.getTexture(),
		int(this->rays),
		this->spaceTimeOptions->samplingRate_0,
		this->raycastingOptions->tolerance_0,
		*spaceTimeOptions,
		*renderingOptions
	);

	
}


__host__ bool FluctuationHeightfield::initializeBoundingBox()
{
	BoundingBox* h_boundingBox = new BoundingBox;

	h_boundingBox->gridSize = make_int3((int)this->m_gridSize3D.x, (int)this->m_gridSize3D.y, (int)this->m_gridSize3D.z); // Use the heightfield dimension instead of velocity volume
	float3 center = Array2Float3(raycastingOptions->clipBoxCenter);
	float3 clipBox = Array2Float3(raycastingOptions->clipBox);

	if (spaceTimeOptions->shifSpaceTime)
	{
		center.x -= spaceTimeOptions->projectionPlanePos;
		//clipBox.x += timeSpaceRenderingOptions->shiftProjectionPlane;
	}

	h_boundingBox->updateBoxFaces(clipBox, center);
	h_boundingBox->gridDiameter = ArrayToFloat3(solverOptions->gridDiameter);
	h_boundingBox->gridDiameter.x = solverOptions->timeDim;
	h_boundingBox->updateAspectRatio(*width, *height);						
	h_boundingBox->constructEyeCoordinates
	(
		XMFloat3ToFloat3(camera->GetPositionFloat3()),
		XMFloat3ToFloat3(camera->GetViewVector()),
		XMFloat3ToFloat3(camera->GetUpVector())
	);					
	h_boundingBox->FOV =static_cast<float>((renderingOptions->FOV_deg / 360.0f) * XM_2PI);
	h_boundingBox->distImagePlane = this->distImagePlane;

	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox_spacetime, h_boundingBox, sizeof(BoundingBox)));
	delete h_boundingBox;

	if (spaceTimeOptions->additionalRaycasting)
	{
		BoundingBox * h_boundingBox = new BoundingBox;

		h_boundingBox->gridSize = ArrayToInt3(solverOptions->gridSize);
		h_boundingBox->updateBoxFaces(ArrayToFloat3(raycastingOptions->clipBox), ArrayToFloat3(raycastingOptions->clipBoxCenter));
		h_boundingBox->updateAspectRatio(*width, *height);
		h_boundingBox->m_eyePos = XMFloat3ToFloat3(camera->GetPositionFloat3());
		h_boundingBox->constructEyeCoordinates
		(
			XMFloat3ToFloat3(camera->GetPositionFloat3()),
			XMFloat3ToFloat3(camera->GetViewVector()),
			XMFloat3ToFloat3(camera->GetUpVector())
		);

		h_boundingBox->FOV = static_cast<float>((this->FOV_deg / 360.0f)* XM_2PI);
		h_boundingBox->distImagePlane = this->distImagePlane;
		h_boundingBox->gridDiameter = ArrayToFloat3(solverOptions->gridDiameter);
		gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));

		delete h_boundingBox;
	}


	return true;
}



__host__ bool FluctuationHeightfield::InitializeHeightSurface3D(cudaArray_t pCudaArray)
{
	// Assign the hightArray to the hightSurface and initialize the surface
	this->s_HeightSurface.setInputArray(pCudaArray);
	if (!this->s_HeightSurface.initializeSurface())
		return false;

	return true;
}


__host__ bool FluctuationHeightfield::loadRaycastingTexture(FieldOptions * fieldOptions,int idx)
{
	volume_IO.Initialize(fieldOptions);
	volume_IO.readVolume_Compressed(idx,Array2Int3(fieldOptions->gridSize));
	volumeTexture3D_height_extra.release();
	float * h_VelocityField = this->volume_IO.getField_float_GPU();

	// Bind and copy device pointer to texture
	volumeTexture3D_height_extra.setField(h_VelocityField);
	volumeTexture3D_height_extra.initialize_devicePointer(Array2Int3(fieldOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);

	return true;
}