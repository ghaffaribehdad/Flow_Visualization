#include "fluctuationHeightfield.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/Cuda_helper_math_host.h"

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
	primary_IO.Initialize(this->fluctuationOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!this->InitializeHeightArray3D_Single(make_int3((int)m_gridSize3D.x, (int)m_gridSize3D.y, (int)m_gridSize3D.z)))
		return false;


	// Trace the fluctuation field
	this->traceFluctuationfield3D();

	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D_Single())
		return false;

	this->gradientFluctuationfield();

	this->s_HeightSurface_Primary.destroySurface();

	if (!this->InitializeHeightTexture3D_Single())
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
		this->primary_IO.readVolumePlane(t + fluctuationOptions->firstIdx, VolumeIO::readPlaneMode::YZ, fluctuationOptions->spanwisePos, offset, buffer_size);
		float* p_temp = primary_IO.getField_float();

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

		primary_IO.release();
	}


	
	this->a_HeightSurface_Primary.memoryCopy(h_velocityField);

	delete[] h_velocityField;
	
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

	fluctuationfieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			s_HeightSurface_Primary.getSurfaceObject(),
			*this->solverOptions,
			*this->fluctuationOptions
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


	CudaTerrainRenderer_extra_fluctuation<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->t_HeightSurface_Primary,
			this->t_HeightSurface_Primary_Ex,
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

__host__ bool FluctuationHeightfield::InitializeHeightTexture3D_Single()
{

	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));


	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = this->a_HeightSurface_Primary.getArray();
	

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;

	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&this->t_HeightSurface_Primary, &resDesc, &texDesc, NULL));



	return true;

}



__host__ bool FluctuationHeightfield::InitializeHeightArray3D_Single(int3 gridSize)
{
	// Set dimensions and initialize height field as a 3D CUDA Array
	this->a_HeightSurface_Primary.setDimension(gridSize.x, gridSize.y, gridSize.z);

	// initialize the 3D array
	if (!a_HeightSurface_Primary.initialize())
		return false;


	return true;
}


__host__ bool FluctuationHeightfield::InitializeHeightSurface3D_Single()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	cudaArray_t pCudaArray = NULL;

	pCudaArray = a_HeightSurface_Primary.getArray();

	this->s_HeightSurface_Primary.setInputArray(pCudaArray);
	if (!this->s_HeightSurface_Primary.initializeSurface())
		return false;


	return true;
}