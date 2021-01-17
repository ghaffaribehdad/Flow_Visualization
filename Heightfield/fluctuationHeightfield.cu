#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "..//Cuda/CudaHelperFunctions.h"

extern __constant__  BoundingBox d_boundingBox;
extern __constant__ float3 d_raycastingColor;

bool FluctuationHeightfield::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->m_gridSize3D =
	{
		(size_t)solverOptions->gridSize[2],
		(size_t)fluctuationheightfieldOptions->wallNormalgridSize,
		(size_t)1 + (size_t)fluctuationheightfieldOptions->lastIdx - (size_t)fluctuationheightfieldOptions->firstIdx

	};


	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	this->rays = (*this->width) * (*this->height);				// Set number of rays based on the number of pixels

	// initialize volume Input Output
	this->volume_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!a_HeightSurface_Primary.initialize(m_gridSize3D.x, m_gridSize3D.y, m_gridSize3D.z))
		return false;

	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D())
		return false;

	// Trace the fluctuation field
	this->traceFluctuationfield3D();

	//this->gradientFluctuationfield();

	this->s_HeightSurface_Primary.destroySurface();

	this->volumeTexture3D_height.setArray(a_HeightSurface_Primary.getArrayRef());
	this->volumeTexture3D_height_extra.setArray(a_HeightSurface_Primary_Extra.getArrayRef());

	this->volumeTexture3D_height.initialize_array();
	this->volumeTexture3D_height.initialize_array();

	return true;
}

void  FluctuationHeightfield::traceFluctuationfield3D()
{

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int nMesh_y = (int)m_gridSize3D.y; // mesh size in spanwise direction

	blocks = static_cast<unsigned int>((nMesh_y % (thread.x * thread.y) == 0 ?
		nMesh_y / (thread.x * thread.y) : nMesh_y / (thread.x * thread.y) + 1));


	for (int t = solverOptions->currentIdx; t <= solverOptions->lastIdx; t++)
	{
		// First Read the Compressed file and move it to GPU
		this->volume_IO.readVolume(t,solverOptions);

		// Copy the device pointer
		float * h_VelocityField = this->volume_IO.getField_float_GPU();

		// Bind and copy device pointer to texture
		volumeTexture.setField(h_VelocityField);

		volumeTexture.initialize_devicePointer(Array2Int3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
		
		// Release the device pointer
		cudaFree(h_VelocityField);

		// Copy from texture to surface
		copyTextureToSurface << < blocks, thread >> >
			(
				10, //		Streamwise Pos
				t, //		Timestep
				solverOptions,	
				volumeTexture.getTexture(),
				s_HeightSurface_Primary.getSurfaceObjectRef()
			);

		// Release the volume texture
		volumeTexture.release();
		
	}
	volume_IO.release();
}



void FluctuationHeightfield::gradientFluctuationfield()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int nMesh_y = (int)m_gridSize3D.y; // mesh size in spanwise direction

	blocks = static_cast<unsigned int>((nMesh_y % (thread.x * thread.y) == 0 ?
		nMesh_y / (thread.x * thread.y) : nMesh_y / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle

	fluctuationfieldGradient3D<FetchTextureSurface::Channel_X> << < blocks, thread >> >
		(
			s_HeightSurface_Primary.getSurfaceObject(),
			*this->solverOptions,
			*this->fluctuationheightfieldOptions
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


	CudaTerrainRenderer_extra_fluctuation<FetchTextureSurface::Channel_X> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->volumeTexture3D_height.getTexture(),
			this->volumeTexture3D_height_extra.getTexture(),
			int(this->rays),
			this->fluctuationheightfieldOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0,
			*fluctuationheightfieldOptions
			);

}


__host__ bool FluctuationHeightfield::initializeBoundingBox()
{
	BoundingBox* h_boundingBox = new BoundingBox;

	h_boundingBox->gridSize = make_int3((int)this->m_gridSize3D.x, (int)this->m_gridSize3D.y, (int)this->m_gridSize3D.z); // Use the heightfield dimension instead of velocity volume
	h_boundingBox->updateBoxFaces(ArrayFloat3ToFloat3(raycastingOptions->clipBox), ArrayFloat3ToFloat3(raycastingOptions->clipBoxCenter));
	h_boundingBox->m_dimensions = ArrayFloat3ToFloat3(solverOptions->gridDiameter);
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



__host__ bool FluctuationHeightfield::InitializeHeightSurface3D()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;

	pCudaArray = a_HeightSurface_Primary.getArray();

	this->s_HeightSurface_Primary.setInputArray(pCudaArray);
	if (!this->s_HeightSurface_Primary.initializeSurface())
		return false;


	return true;
}