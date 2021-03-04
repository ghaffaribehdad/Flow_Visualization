#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "..//Cuda/CudaHelperFunctions.h"

extern __constant__  BoundingBox d_boundingBox;

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
	this->volume_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!a_HeightArray3D.initialize(m_gridSize3D.x, m_gridSize3D.y, m_gridSize3D.z))
		return false;


	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D(a_HeightArray3D.getArray()))
		return false;

	// Trace the fluctuation field
	this->generateTimeSpaceField3D(timeSpaceRenderingOptions);

	// Destroy the surface
	this->s_HeightSurface.destroySurface();



	switch (timeSpaceRenderingOptions->gaussianFilter)
	{
	case true: // Filter
	{
		// Initilize volume texture to do filtering
		this->volumeTexture3D_height.setArray(a_HeightArray3D.getArrayRef());
		this->volumeTexture3D_height.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

		// Allocate new 3D Array
		if (!a_HeightArray3D_Copy.initialize(m_gridSize3D.x, m_gridSize3D.y, m_gridSize3D.z))
			return false;

		// Bind the new array 3D to cuda surface and initilize it
		cudaArray_t pCudaArray = a_HeightArray3D_Copy.getArray();
		this->s_HeightSurface.setInputArray(pCudaArray);
		if (!this->s_HeightSurface.initializeSurface())
			return false;

		this->gaussianFilter();

		// Release the unfiltered (destroy array)
		this->volumeTexture3D_height.release();
		this->s_HeightSurface.destroySurface();

		volumeTexture3D_height.setArray(a_HeightArray3D_Copy.getArrayRef());

		break;
	}
	case false: // No Filter
	{
		volumeTexture3D_height.setArray(a_HeightArray3D.getArrayRef());
		break;
	}

	}



	this->volumeTexture3D_height.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);


	return true;
}


void  FluctuationHeightfield::generateTimeSpaceField3D(TimeSpaceRenderingOptions * timeSpaceOptions)
{

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	blocks = BLOCK_THREAD(m_gridSize3D.z); // Kernel calls are based on the Spanwise gridSize

	
	for (int t = 0; t < m_gridSize3D.x; t++) // Iterates over time
	{
		// First Read the Compressed file and move it to GPU
		this->volume_IO.readVolume_Compressed(t+solverOptions->firstIdx, Array2Int3(solverOptions->gridSize));

		// Copy the device pointer
		float * h_VelocityField = this->volume_IO.getField_float_GPU();

		// Bind and copy device pointer to texture
		volumeTexture.setField(h_VelocityField);

		volumeTexture.initialize_devicePointer(Array2Int3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
		
		// Release the device pointer
		cudaFree(h_VelocityField);
		
		// Copy from texture to surface
		copyTextureToSurface<FetchTextureSurface::Channel_X, FetchTextureSurface::Channel_Y, FetchTextureSurface::Channel_Z, FetchTextureSurface::Channel_W> << < blocks, thread >> >
			(
				solverOptions->projectPos, //		Streamwise Pos
				t, //		Timestep
				*solverOptions,	
				volumeTexture.getTexture(),
				s_HeightSurface.getSurfaceObject()
			);

		// Release the volume texture
		volumeTexture.release();
		
	}
	volume_IO.release();
}



void FluctuationHeightfield::gaussianFilter()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	blocks = BLOCK_THREAD((int)m_gridSize3D.y);

	applyGaussianFilter << < blocks, thread >> > 
	(
		9,
		{ m_gridSize3D.x,m_gridSize3D.y,m_gridSize3D.z},
		this->volumeTexture3D_height.getTexture(),
		this->s_HeightSurface.getSurfaceObject()
	);


}

__host__ void FluctuationHeightfield::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Create a 2D texture to read hight array


	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));

	timeSpaceRenderingOptions->currentTime = solverOptions->currentIdx - solverOptions->firstIdx ;



	CudaTerrainRenderer_extra_fluctuation<FetchTextureSurface::Channel_X, FetchTextureSurface::Channel_Y> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->volumeTexture3D_height.getTexture(),
			int(this->rays),
			this->timeSpaceRenderingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0,
			*timeSpaceRenderingOptions
			);

}


__host__ bool FluctuationHeightfield::initializeBoundingBox()
{
	BoundingBox* h_boundingBox = new BoundingBox;

	h_boundingBox->gridSize = make_int3((int)this->m_gridSize3D.x, (int)this->m_gridSize3D.y, (int)this->m_gridSize3D.z); // Use the heightfield dimension instead of velocity volume
	h_boundingBox->updateBoxFaces(ArrayFloat3ToFloat3(raycastingOptions->clipBox), ArrayFloat3ToFloat3(raycastingOptions->clipBoxCenter));
	h_boundingBox->m_dimensions = ArrayFloat3ToFloat3(solverOptions->gridDiameter);
	h_boundingBox->m_dimensions.x = solverOptions->timeDim;
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


	delete h_boundingBox;

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