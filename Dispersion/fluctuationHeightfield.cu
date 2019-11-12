#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"

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
		(size_t)fluctuationOptions->wallNormalgridSize,
		(size_t)1 + (size_t)fluctuationOptions->lastIdx - (size_t)fluctuationOptions->firstIdx

	};


	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	this->rays = (*this->width) * (*this->height);				// Set number of rays based on the number of pixels

	// initialize volume Input Output
	volume_IO.Initialize(this->fluctuationOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!this->InitializeHeightArray3D(make_int3((int)m_gridSize3D.x, (int)m_gridSize3D.y, (int)m_gridSize3D.z)))
		return false;


	// Trace the fluctuation field
	this->traceFluctuationfield3D();

	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D())
		return false;

	this->gradientFluctuationfield();

	this->heightSurface3D.destroySurface();

	if (!this->InitializeHeightTexture3D())
		return false;


	return true;
}

void  FluctuationHeightfield::traceFluctuationfield3D()
{
	float* h_velocityField = new float[m_gridSize3D.x * m_gridSize3D.y * m_gridSize3D.z * (size_t)4];

	size_t offset = solverOptions->gridSize[2] * solverOptions->gridSize[1] * sizeof(float4);
	size_t buffer_size = solverOptions->gridSize[2] * (size_t)fluctuationOptions->wallNormalgridSize * sizeof(float4);
	


	size_t counter = 0;

	for (int t = 0; t < m_gridSize3D.z; t++)
	{
		this->volume_IO.readVolumePlane(t + fluctuationOptions->firstIdx, volumeIO::readPlaneMode::YZ, fluctuationOptions->spanwisePos, offset, buffer_size);
		float* p_temp = volume_IO.flushBuffer_float();

		size_t counter_t = 0;


		for (int wall = 0; wall < m_gridSize3D.y; wall++)
		{

			for (int span = 0; span < m_gridSize3D.x; span++)
			{
				for (int d = 0; d < 4; d++)
				{

					h_velocityField[counter] = p_temp[counter_t];
					counter++;
					counter_t++;

				}

			}
		}

		volume_IO.release();
	}


	
	this->heightArray3D.memoryCopy(h_velocityField);

	delete[] h_velocityField;
	
}



void FluctuationHeightfield::gradientFluctuationfield()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int nMesh_x = (int)m_gridSize3D.x; // mesh size in spanwise direction

	blocks = static_cast<unsigned int>((nMesh_x % (thread.x * thread.y) == 0 ?
		nMesh_x / (thread.x * thread.y) : nMesh_x / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle

	fluctuationfieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			heightSurface3D.getSurfaceObject(),
			*this->solverOptions,
			*this->fluctuationOptions
		);
}



__host__ void FluctuationHeightfield::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Create a 2D texture to read hight array

	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));


	CudaTerrainRenderer_extra_fluctuation<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->heightFieldTexture3D,
			this->heightFieldTexture3D_extra,
			int(this->rays),
			this->fluctuationOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0,
			*fluctuationOptions
			);

}


__host__ bool FluctuationHeightfield::initializeBoundingBox()
{
	BoundingBox* h_boundingBox = new BoundingBox;

	h_boundingBox->gridSize = make_int3((int)this->m_gridSize3D.x, (int)this->m_gridSize3D.y, (int)this->m_gridSize3D.z); // Use the heightfield dimension instead of velocity volume
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

__host__ bool FluctuationHeightfield::InitializeHeightTexture3D()
{

	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));


	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->heightArray3D.getArray();
	

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->heightFieldTexture3D, &resDesc, &texDesc, NULL));



	return true;

}



__host__ bool FluctuationHeightfield::InitializeHeightArray3D(int3 gridSize)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->heightArray3D.setDimension(gridSize.x, gridSize.y, gridSize.z);

	// initialize the 3D array
	if (!heightArray3D.initialize())
		return false;


	return true;
}


__host__ bool FluctuationHeightfield::InitializeHeightSurface3D()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;

	pCudaArray = heightArray3D.getArray();

	this->heightSurface3D.setInputArray(pCudaArray);
	if (!this->heightSurface3D.initializeSurface())
		return false;


	return true;
}