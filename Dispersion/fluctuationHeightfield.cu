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
	this->m_gridSize = 
	{
		solverOptions->gridSize[2],								//=> spanwise(z)
		solverOptions->gridSize[1],								//=> # time snaps
		1 + solverOptions->lastIdx - solverOptions->firstIdx	//=> wall-normal grid size (+1 for inclusive indexes)
	};



	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	this->rays = (*this->width) * (*this->height);				// Set number of rays based on the number of pixels

	// initialize volume Input Output
	volume_IO.Initialize(this->solverOptions);


	// Initialize Height Field as an empty CUDA array 3D
	if (!this->InitializeHeightArray3D(m_gridSize))
		return false;

	// Bind the array of heights to the CUDA surface
	if (!this->InitializeHeightSurface3D())
		return false;

	// Trace the fluctuation field
	this->traceFluctuationfield();

	this->gradientFluctuationfield();

	this->heightSurface3D.destroySurface();

	if (!this->InitializeHeightTexture3D())
		return false;


	return true;
}

void  FluctuationHeightfield::traceFluctuationfield()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	int nMesh_z = solverOptions->gridSize[2]; // mesh size in spanwise direction

	blocks = static_cast<unsigned int>((nMesh_z % (thread.x * thread.y) == 0 ?
		nMesh_z / (thread.x * thread.y) : nMesh_z / (thread.x * thread.y) + 1));

	// Set properties of velocity field texture

	int timeDimension = 0;

	for (int timestep = solverOptions->firstIdx; timestep <= solverOptions->lastIdx; timestep++)
	{
		// loads the velocity field into the host memory (accessible by this->field pointer)  !!!! only one plane
		this->LoadVelocityfieldPlane(timestep,solverOptions->gridSize[0]/2);
		this->velocityField_2D.setField(this->field);


		// Copy host memory into the device texture (using border mode since no advection desired here!)
		velocityField_2D.initialize(solverOptions->gridSize[1], solverOptions->gridSize[2]);

		// Release the volume in the host memory
		this->volume_IO.release();

		// Now is time to populate the texture
		trace_fluctuation3D << < blocks, thread >> >
			(
				heightSurface3D.getSurfaceObject(),
				heightSurface3D_extra.getSurfaceObject(),
				this->velocityField_2D.getTexture(),
				*solverOptions,
				*dispersionOptions,
				timeDimension,
				0.5f //=> Streamwise position
			);
		velocityField_2D.release();

		// Go to next timestep
		timeDimension++;

	}

}


bool FluctuationHeightfield::LoadVelocityfieldPlane(const unsigned int& idx, const int& plane)
{
	if (!volume_IO.readVolumePlane(idx,volumeIO::readPlaneMode::YZ,plane))
		return false;

	this->field = volume_IO.flushBuffer_float();

	return true;
}




void FluctuationHeightfield::gradientFluctuationfield()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	int nMesh_z = solverOptions->gridSize[2]; // mesh size in spanwise direction

	blocks = static_cast<unsigned int>((nMesh_z % (thread.x * thread.y) == 0 ?
		nMesh_z / (thread.x * thread.y) : nMesh_z / (thread.x * thread.y) + 1));

	// After this step the heightSurface is populated with the height of each particle

	fluctuationfieldGradient3D<IsosurfaceHelper::Position> << < blocks, thread >> >
		(
			heightSurface3D.getSurfaceObject(),
			*this->solverOptions
		);
}



__host__ void FluctuationHeightfield::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Create a 2D texture tu read hight array

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
			this->raycastingOptions->samplingRate_0,
			this->raycastingOptions->tolerance_0,
			*fluctuationOptions
			);

}


__host__ bool FluctuationHeightfield::initializeBoundingBox()
{
	BoundingBox* h_boundingBox = new BoundingBox;

	h_boundingBox->gridSize = this->m_gridSize; // Use the heightfield dimension instead of velocity volume
	h_boundingBox->updateBoxFaces(ArrayFloat3ToFloat3(solverOptions->gridDiameter)); // what if we use this??
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

