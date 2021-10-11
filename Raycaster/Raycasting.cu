#include "Raycasting.h"
#include "IsosurfaceHelperFunctions.h"
#include "Raycasting_Helper.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Cuda/helper_math.h"
#include "..//Cuda/Cuda_helper_math_host.h"


__constant__	BoundingBox d_boundingBox; // constant memory variable
__constant__	BoundingBox d_boundingBox_spacetime;






// Explicit instantiation
template __global__ void CudaIsoSurfacRendererSpaceTime<struct FetchTextureSurface::Channel_X>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer_GradientBase< struct FetchTextureSurface::Lambda2 >		(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, RaycastingOptions raycastingOptions);
template __global__ void CudaTerrainRenderer_extra< struct FetchTextureSurface::Channel_X >				(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t extraField, int rays, float samplingRate, float IsosurfaceTolerance, DispersionOptions dispersionOptions, int traceTime);
template __global__ void CudaTerrainRenderer_extra_fluctuation< struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_Y >	(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation <struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_Z>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation <struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_X>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation_raycasting <struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_Y>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t isoSurface,int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions, RaycastingOptions raycastingOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation_raycasting <struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_X>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t isoSurface,int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions, RaycastingOptions raycastingOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation_raycasting <struct FetchTextureSurface::Channel_X, struct FetchTextureSurface::Channel_Z>(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t isoSurface,int rays, float samplingRate, float IsosurfaceTolerance, SpaceTimeOptions fluctuationOptions, RaycastingOptions raycastingOptions);

__host__ bool Raycasting::updateScene()
{
	if (!this->initializeRaycastingInteroperability())	// Create interoperability while we need to release it at the end of rendering
		return false;

	if (!this->initializeCudaSurface())					// initialize cudaSurface	
		return false;

	if (!this->initializeBoundingBox())					//updates constant memory
		return false;

	this->rendering();


	if (!this->raycastingSurface.destroySurface())
		return false;

	this->interoperatibility.release();

	return true;

}

void Raycasting::loadTexture
(
	SolverOptions * solverOptions,
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ
)
{
	// Read current volume
	this->volume_IO_Primary.readVolume(idx);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO_Primary.getField_float();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	volumeTexture.initialize(Array2Int3(solverOptions->gridSize), false, addressModeX, addressModeY, addressModeZ);
	// release host memory
	volume_IO_Primary.release();
}

void Raycasting::loadTexture
(
	int3 & gridSize,
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ
)
{
	// Read current volume
	this->volume_IO_Primary.readVolume(idx);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO_Primary.getField_float();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	volumeTexture.initialize(gridSize, false, addressModeX, addressModeY, addressModeZ);
	// release host memory
	volume_IO_Primary.release();
}


void Raycasting::loadTextureCompressed
(
	int3 & gridSize,
	VolumeTexture3D & volumeTexture,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ)
{

	Timer timer;

	// Read current volume
	this->volume_IO_Primary.readVolume_Compressed(idx, gridSize);
	// Return a pointer to volume
	float * h_VelocityField = this->volume_IO_Primary.getField_float_GPU();
	// set the pointer to the volume texture
	volumeTexture.setField(h_VelocityField);
	// initialize the volume texture
	TIMELAPSE(volumeTexture.initialize_devicePointer(gridSize, false, addressModeX, addressModeY, addressModeZ);, "Initialize Texture including DDCopy");

	cudaFree(h_VelocityField);
}


void Raycasting::loadTextureCompressed_double
(
	int * gridSize0,
	int * gridSize1,
	VolumeTexture3D & volumeTexture_0,
	VolumeTexture3D & volumeTexture_1,
	const int & idx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ
)
{

	Timer timer;

	// Read current volume
	this->volume_IO_Primary.readVolume_Compressed(idx, Array2Int3(gridSize0));
	float * h_VelocityField = this->volume_IO_Primary.getField_float_GPU();
	// set the pointer to the volume texture
	volumeTexture_0.setField(h_VelocityField);
	// initialize the volume texture
	TIMELAPSE(volumeTexture_0.initialize_devicePointer(Array2Int3(gridSize0), false, addressModeX, addressModeY, addressModeZ); , "Initialize Texture including DDCopy");
	// Free host memory
	cudaFree(h_VelocityField);


	//Same procedure for the secondary dataset
	this->volume_IO_Secondary.readVolume_Compressed(idx, Array2Int3(gridSize1));
	h_VelocityField = this->volume_IO_Secondary.getField_float_GPU();
	volumeTexture_1.setField(h_VelocityField);
	TIMELAPSE(volumeTexture_1.initialize_devicePointer(Array2Int3(gridSize1), false, addressModeX, addressModeY, addressModeZ);, "Initialize Texture including DDCopy");
	cudaFree(h_VelocityField);
}



__host__ bool Raycasting::initialize_Single
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z 
)
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

	this->volume_IO_Primary.Initialize(this->solverOptions);

	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
	{
		int3 gridSize = Array2Int3(solverOptions->gridSize);
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);

		break;
	}

	case false: // Uncompressed Data
	{
		loadTexture(solverOptions, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	}

	
	this->raycastingOptions->fileLoaded = true;
	this->b_initialized = true;

	return true;

}



__host__ bool Raycasting::initialize_Double
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;

	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels

	
	this->volume_IO_Primary.Initialize(this->solverOptions);
	this->volume_IO_Secondary.Initialize(&fieldOptions[1]);


	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
	{
		int3 gridSize = ArrayInt2ToInt3(fieldOptions[0].gridSize);
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		
		gridSize = Array2Int3(fieldOptions[1].gridSize);
		loadTextureCompressed(gridSize, this->volumeTexture_1, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	case false: // Uncompressed Data
	{
		int3 gridSize = ArrayInt2ToInt3(fieldOptions[0].gridSize);
		loadTexture(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);

		gridSize = Array2Int3(fieldOptions[1].gridSize);
		loadTexture(gridSize, this->volumeTexture_1, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	}

	this->raycastingOptions->fileLoaded = true;
	this->b_initialized = true;

	return true;

}


__host__ bool Raycasting::initialize_Multiscale
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;

	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels
	this->volume_IO_Primary.Initialize(this->solverOptions);
	int3 gridSize = Array2Int3(fieldOptions[0].gridSize);

	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
		
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;

	case false: // Uncompressed Data
	
		loadTexture(solverOptions,this->volumeTexture_0,solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;

	}


	initializeMipmapL1();
	generateMipmapL1();
	volumeTexture_L1.setArray(a_mipmap_L1.getArrayRef());
	volumeTexture_L1.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	initializeMipmapL2();
	generateMipmapL2();
	volumeTexture_L2.setArray(a_mipmap_L2.getArrayRef());
	volumeTexture_L2.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	this->raycastingOptions->fileLoaded = true;
	this->b_initialized = true;

	return true;

}




__host__ bool Raycasting::release()
{
	this->volumeTexture_0.release();
	this->volumeTexture_L1.release();
	this->volumeTexture_L2.release();

	this->volume_IO_Primary.release();
	this->volume_IO_Secondary.release();

	this->a_mipmap_L1.release();
	this->a_mipmap_L2.release();
	
	this->b_initialized = false;
	return true;
}


void Raycasting::updateFile_Single
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();

	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
	{

		int3 gridSize = Array2Int3(solverOptions->gridSize);
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);

		break;
	}

	case false: // Uncompressed Data
	{
		loadTexture(solverOptions, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	}
	this->raycastingOptions->fileChanged = false;
}



void Raycasting::updateFile_Mult
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
	int3 gridSize = Array2Int3(fieldOptions[0].gridSize);

	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
		
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);

		gridSize = Array2Int3(fieldOptions[1].gridSize);
		loadTextureCompressed(gridSize, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	
	case false: // Uncompressed Data
		loadTexture(solverOptions, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	this->raycastingOptions->fileChanged = false;
}




void Raycasting::updateFile_MultiScale
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
	this->volumeTexture_L1.release();
	this->volumeTexture_L2.release();


	switch (solverOptions->compressed)
	{

	case true: // Compressed Data
	{
		loadTextureCompressed_double(fieldOptions[1].gridSize, fieldOptions[1].gridSize, this->volumeTexture_0, this->volumeTexture_1, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);

		break;
	}

	case false: // Uncompressed Data
	{
		loadTexture(solverOptions, this->volumeTexture_0, solverOptions->currentIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}

	}



	generateMipmapL1();
	volumeTexture_L1.setArray(a_mipmap_L1.getArrayRef());
	volumeTexture_L1.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	generateMipmapL2();
	volumeTexture_L2.setArray(a_mipmap_L2.getArrayRef());
	volumeTexture_L2.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	this->raycastingOptions->fileChanged = false;
}

__host__ void Raycasting::rendering()
{

	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(rays);
	int3 gridSize = Array2Int3(solverOptions->gridSize);



	switch (raycastingOptions->raycastingMode)
	{
	case RaycastingMode::Mode::SINGLE:

		CudaIsoSurfacRenderer_Single << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions
				);

		break;

	case RaycastingMode::Mode::DOUBLE:


		CudaIsoSurfacRenderer_Double << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions
				);
		break;

	case RaycastingMode::Mode::MULTISCALE:

		CudaIsoSurfacRenderer_Single << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions
				);
		break;
	}





	//// TODO:
	//// Alternatively use ENUM templates! 
	//switch (this->raycastingOptions->isoMeasure_0)
	//{
	//case IsoMeasure::VelocityMagnitude:
	//{
	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::Velocity_Magnitude> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}
	//	break;

	//}

	//case IsoMeasure::Velocity_X:
	//{
	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::Channel_X> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}


	//	break;

	//}

	//case IsoMeasure::Velocity_Y:
	//{
	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::Channel_Y> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}

	//	break;
	//}

	//case IsoMeasure::Velocity_Z:
	//{

	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::Channel_Z> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}


	//	break;
	//}


	//case IsoMeasure::Velocity_W:
	//{

	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::Channel_W> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}

	//	break;
	//}


	//case IsoMeasure::ShearStress:
	//{
	//	if (raycastingOptions->planarRaycasting)
	//	{
	//		CudaIsoSurfacRenderer_float_PlaneColor<FetchTextureSurface::ShearStress> << < blocks, thread >> >
	//			(
	//				raycastingSurface.getSurfaceObject(),
	//				this->volumeTexture_0.getTexture(),
	//				int(this->rays),
	//				gridSize,
	//				*raycastingOptions,
	//				*this->solverOptions
	//				);
	//	}
	//	else


	//	break;
	//}


	//case::IsoMeasure::LAMBDA2:
	//{
	//	CudaIsoSurfacRenderer_GradientBase<FetchTextureSurface::Lambda2> << < blocks, thread >> >
	//		(
	//			this->raycastingSurface.getSurfaceObject(),
	//			this->volumeTexture_0.getTexture(),
	//			int(this->rays),
	//			*this->raycastingOptions
	//			);
	//	break;
	//}


	//}



}



__host__ bool Raycasting::initializeBoundingBox()
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
	

	h_boundingBox->FOV = (this->FOV_deg / 360.0f)* XM_2PI;
	h_boundingBox->distImagePlane = this->distImagePlane;
	h_boundingBox->m_dimensions = ArrayToFloat3(solverOptions->gridDiameter);
	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));
	//if (!gpuErrchk(cudaMemcpyToSymbol(reinterpret_cast<const void *>(d_boundingBox), h_boundingBox, sizeof(BoundingBox))))
	//{
	//	ErrorLogger::Log("Fail to Copy to Constant Memory");
	//	return false;
	//}

	delete h_boundingBox;
	
	return true;
}



__global__ void CudaIsoSurfacRenderer_TurbulentDiffusivity
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	cudaTextureObject_t t_avg_temp,
	int rays, float isoValue,
	float samplingRate,
	float IsosurfaceTolerance,
	float * avg_temp
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{
		int2 pixel = index2pixel(index, d_boundingBox.m_width);				// determine pixel position based on the index of the thread
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);	// Find the position of pixel on the view plane
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);		// find Enter and Exit point of the Ray

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = (position / d_boundingBox.m_dimensions);
				FetchTextureSurface::TurbulentDiffusivity turbulentDiff;

				float value = turbulentDiff.ValueAtXYZ_avgtemp(field1, relativePos, d_boundingBox.gridSize, t_avg_temp);
				float3 dir = rayDir * t ;

				// check if we have a hit 
				if (value > isoValue)
				{

					position = turbulentDiff.binarySearch_avgtemp(
							field1,
							t_avg_temp,
							d_boundingBox.gridSize,
							position,
							d_boundingBox.m_dimensions,
							dir,
							isoValue,
							IsosurfaceTolerance, 
							50
						);
					relativePos = (position / d_boundingBox.m_dimensions);

					// calculates gradient
					float3 gradient = turbulentDiff.GradientAtGrid_AvgTemp(field1, relativePos, d_boundingBox.gridSize, t_avg_temp);

					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);

					float3 rgb = make_float3(1,1,1) * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;
				}


			}


		}

	}


}

template <typename Observable>
__global__ void CudaIsoSurfacRendererAnalytic
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions
)
{

	Observable observable;
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.m_dimensions / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		//float2 NearFar = findIntersections(pixelPos, d_boundingBox);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				float value = observable.ValueAtXYZ_Tex(field1, relativePos);
				float2 tEnterExit = { t,findExitPoint3D(position, dir, cellSize) };

				//float4 factors = getFactors<float>(values, position, dir);
				// check if we have a hit 
				if (value > raycastingOptions.isoValue_0)
				{

					//position = binarySearch<Observable>(observable, field1, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, dir * t, raycastingOptions.isoValue_0, raycastingOptions.tolerance_0, 200);
					relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

					// calculates gradient
					float3 gradient = observable.GradientAtXYZ_Tex(field1, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_0) ^ raycastingOptions.brightness;
					float3 rgb = raycastingColor * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;
				}

				float gridStep = findExitPoint3D(position, dir, cellSize);
				if (raycastingOptions.adaptiveSampling)
				{
					t += gridStep;
				}
				else
				{
					t = t + raycastingOptions.samplingRate_0;
				}
			}


		}

	}


}




template <typename Observable>
__global__ void CudaIsoSurfacRenderer_GradientBase
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions
)
{

	Observable observable;

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);

		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + raycastingOptions.samplingRate_0)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				float3 box = relativePos / d_boundingBox.gridSize;
		   
					// check if we have a hit 
				if (observable.ValueAtXYZ_Tex_GradientBase(field1, relativePos,d_boundingBox.m_dimensions,d_boundingBox.gridSize) > raycastingOptions.isoValue_0)
				{


					// calculates gradient
					float3 gradient = observable.GradientAtXYZ_Tex_GradientBase(field1, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_0);
					float3 rgb = raycastingColor * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;
				}
			}


		}

	}


}



template <typename Observable>
__global__ void CudaTerrainRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
)
{
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);

		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = 
				{ 
					position.x / d_boundingBox.m_dimensions.x,
					position.z / d_boundingBox.m_dimensions.z,
					static_cast<float> (dispersionOptions.timestep) / static_cast<float> (traceTime)
				};
				
				// fetch texels from the GPU memory
				float4 hightFieldVal = ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 &&  position.y - hightFieldVal.x < 0.01 )
				{

					hightFieldVal = ValueAtXYZ_float4(heightField, relativePos);

					float3 gradient = { -hightFieldVal.y,-1,-hightFieldVal.z };


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);

					float3 rgb = { 0,1,0 };


					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}


			}

		}

	}


}


__global__ void CudaTerrainRenderer_extra_FTLE
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
)
{

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;
				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos =
				{
					position.x / d_boundingBox.m_dimensions.x,
					position.z / d_boundingBox.m_dimensions.z,
					static_cast<float> (dispersionOptions.timestep) / static_cast<float> (traceTime)
				};

				// fetch texels from the GPU memory
				float4 hightFieldVal = ValueAtXYZ_Texture_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

					
					float3 gradient = { hightFieldVal.y,-1,hightFieldVal.z };
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);

					float ftle = hightFieldVal.w;

					// shading (no ambient)

					float3 rgb_min = make_float3(dispersionOptions.minColor[0], dispersionOptions.minColor[1], dispersionOptions.minColor[2]);
					float3 rgb_max = make_float3(dispersionOptions.maxColor[0], dispersionOptions.maxColor[1], dispersionOptions.maxColor[2]);

					float extractedVal = saturate((ftle - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

					float3 rgb = rgb_min * (1 - extractedVal) + (extractedVal * rgb_max);

					rgb = rgb * diffuse;

					//// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					//// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					//// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
					

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}


			}



		}

	}
}



template <typename Observable>
__global__ void CudaTerrainRenderer_extra
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos =
				{
					position.z / d_boundingBox.m_dimensions.z,
					position.x / d_boundingBox.m_dimensions.x,
					static_cast<float> (dispersionOptions.timestep) / static_cast<float> (traceTime)
				};

				// fetch texels from the GPU memory
				float4 hightFieldVal = ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

		
					hightFieldVal = ValueAtXYZ_float4(heightField, relativePos);
					
					float3 gradient = { hightFieldVal.y,-1,hightFieldVal.z };


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);

					
					float3 rgb_min = 
					{ 
						dispersionOptions.minColor[0],
						dispersionOptions.minColor[1],
						dispersionOptions.minColor[2],
					};

					float3 rgb_max =
					{
						dispersionOptions.maxColor[0],
						dispersionOptions.maxColor[1],
						dispersionOptions.maxColor[2],
					};

					float3 rgb = rgb_min;

					float extractedVal = 0.0f;
					float3 rgb_complement = { 0.0f,0.0f,0.0f };

					switch (dispersionOptions.colorCode)
					{
						case dispersionOptionsMode::dispersionColorCode::NONE:
						{
							break;
						}

						case dispersionOptionsMode::dispersionColorCode::V_X_FLUCTUATION:
						{

							float4 value = ValueAtXYZ_float4(extraField, relativePos);
							extractedVal = value.y;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::V_Y:
						{

							extractedVal = ValueAtXYZ_float4(extraField, relativePos).z;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::V_Z:
						{

							extractedVal = ValueAtXYZ_float4(extraField, relativePos).w;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DISTANCE:
						{
							float4 Vec_0 = ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y,0.0));
							float4 Vec_1 = ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y,0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { Vec_0.w,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DISTANCE_ZY:
						{
							float4 Vec_0 = ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { 0.0f,Vec_0.x,Vec_1.x };

							Vec_0 = ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { 0.0f,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DEV_Z:
						{
							float4 Vec_1 = ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float position_0 = Vec_1.x;

							Vec_1 = ValueAtXYZ_float4(extraField, relativePos);

							float position_1 = Vec_1.x;

							extractedVal = position_1 - position_0;
							rgb_complement = { 0,0,0 };

							if (extractedVal < 0)
							{
								extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (-dispersionOptions.min_val));
								rgb_complement = make_float3(1, 1, 1) - rgb_min;
								rgb = rgb_complement * extractedVal + rgb_min;
							}
							else
							{
								extractedVal = saturate((dispersionOptions.max_val - extractedVal) / (dispersionOptions.max_val));
								rgb_complement = make_float3(1, 1, 1) - rgb_max;
								rgb = rgb_complement * extractedVal + rgb_max;

							}


						}
						case dispersionOptionsMode::dispersionColorCode::QUADRANT_DEV:
						{
							float4 Vec_0 = ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { Vec_0.w,Vec_0.x,Vec_1.x };

							float2 dev_XZ =
							{
								(position_1.x - position_0.x),
								position_1.z - position_0.z
							};

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							float3 color1 = {	228.0f	/	255.0f,		26.0f	/ 255.0f,	28.0f	/ 255.0f };
							float3 color2 = {	55.0f	/	255.0f,		126.0f	/ 255.0f,	184.0f	/ 255.0f };
							float3 color3 = {	77.0f	/	255.0f,		175.0f	/ 255.0f,	74.0f	/ 255.0f };
							float3 color4 = {	152.0f	/	255.0f,		78.0f	/ 255.0f,	163.0f	/ 255.0f };

							if (dev_XZ.x < 0)
							{
								if (dev_XZ.y > 0)
								{
									rgb_complement = make_float3(1, 1, 1) - color1;
									rgb = rgb_complement * (1.0f - extractedVal) + color1;
								}
								else
								{
									rgb_complement = make_float3(1, 1, 1) - color2;
									rgb = rgb_complement * (1.0f - extractedVal) + color2;
								}

							}
							else
							{
								if (dev_XZ.y > 0)
								{
									rgb_complement = make_float3(1, 1, 1) - color3;
									rgb = rgb_complement * (1.0f - extractedVal) + color3;
								}
								else
								{
									rgb_complement = make_float3(1, 1, 1) - color4;
									rgb = rgb_complement * (1.0f - extractedVal) + color4;

								}


							}

							break;
						}



						case dispersionOptionsMode::dispersionColorCode::DEV_ZY:
						{
							float4 Vec_0 = ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { Vec_0.w,Vec_0.x,Vec_1.x };



							float delta_x = (position_1.x - position_0.x)/10;
							float delta_z = position_1.z - position_0.z;
							
							float delta_x_col = saturate((delta_x - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							float delta_z_col = saturate((delta_z - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							float3 color1 = {1,		0,		0 };
							float3 color2 = {0,		0,	1 };


							rgb_complement = make_float3(1, 1, 1) - color1;
							rgb = rgb_complement * (1.0f - delta_x_col) + color1;

							rgb_complement = make_float3(1, 1, 1) - color2;
							rgb += rgb_complement * (1.0f - delta_z_col) + color2;

							break;
						}
					}

					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}


			}



		}

	}


}

bool Raycasting::initializeRaycastingTexture()
{
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));

	textureDesc.ArraySize = 1;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	textureDesc.Height = *this->height;
	textureDesc.Width = *this->width;
	textureDesc.MipLevels = 1;
	textureDesc.MiscFlags = 0;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;


	HRESULT hr = this->device->CreateTexture2D(&textureDesc, nullptr, this->raycastingTexture.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Front Texture");
	}

	// Create Render targe view
	hr = this->device->CreateRenderTargetView(raycastingTexture.Get(), NULL, this->renderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create RenderTargetView");
		return false;
	}

	return true;

}

bool Raycasting::initializeRaycastingInteroperability()
{
	// define interoperation descriptor and set it to zero
	Interoperability_desc interoperability_desc;
	memset(&interoperability_desc, 0, sizeof(interoperability_desc));

	// set interoperation descriptor
	interoperability_desc.flag = cudaGraphicsRegisterFlagsSurfaceLoadStore;
	interoperability_desc.p_adapter = this->pAdapter;
	interoperability_desc.p_device = this->device;
	//interoperability_desc.size = sizeof(float) * static_cast<size_t>(*this->width) * static_cast<size_t>(*this->height);
	interoperability_desc.size = (size_t)4.0 * sizeof(float) * static_cast<size_t>(*this->width) * static_cast<size_t>(*this->height);
	interoperability_desc.pD3DResource = this->raycastingTexture.Get();

	// initialize the interoperation
	this->interoperatibility.setInteroperability_desc(interoperability_desc);

	return this->interoperatibility.Initialize();
}



__host__ bool Raycasting::initializeCudaSurface()
{

	cudaArray_t pCudaArray = NULL;


	// Get the cuda Array from the interoperability ( array to the texture)
	this->interoperatibility.getMappedArray(pCudaArray);

	// Pass this cuda Array to the raycasting Surface
	this->raycastingSurface.setInputArray(pCudaArray);

	//this->raycastingSurface.setDimensions(*this->width, *this->height);

	// Create cuda surface 
	if (!this->raycastingSurface.initializeSurface())
		return false;

	// To release we need to destroy surface and free the CUDA array kept in the interpoly

	return true;
}


__host__ void Raycasting::setResources
(
	Camera* _camera,
	int* _width,
	int* _height,
	SolverOptions* _solverOption,
	RaycastingOptions* _raycastingOptions,
	RenderingOptions* _renderingOptions,
	ID3D11Device* _device,
	IDXGIAdapter* _pAdapter,
	ID3D11DeviceContext* _deviceContext
)
{
	this->camera = _camera;
	this->FOV_deg = 30.0;
	this->width = _width;
	this->height = _height;

	this->solverOptions = _solverOption;
	this->raycastingOptions = _raycastingOptions;
	this->renderingOptions = _renderingOptions;

	this->device = _device;
	this->pAdapter = _pAdapter;
	this->deviceContext = _deviceContext;

}


__host__ void Raycasting::setResources
(
	Camera* _camera,
	int* _width,
	int* _height,
	SolverOptions* _solverOption,
	RaycastingOptions* _raycastingOptions,
	RenderingOptions* _renderingOptions,
	ID3D11Device* _device,
	IDXGIAdapter* _pAdapter,
	ID3D11DeviceContext* _deviceContext,
	FieldOptions * _fieldOptions
)
{
	this->camera = _camera;
	this->FOV_deg = 30.0;
	this->width = _width;
	this->height = _height;

	this->solverOptions = _solverOption;
	this->raycastingOptions = _raycastingOptions;
	this->renderingOptions = _renderingOptions;

	this->device = _device;
	this->pAdapter = _pAdapter;
	this->deviceContext = _deviceContext;
	this->fieldOptions = _fieldOptions;

}


bool Raycasting::initializeShaders()
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

		if (!pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderTextureSampler.cso"))
			return false;
	}


	return true;
}


bool Raycasting::initializeScene()
{
	if (vertexBuffer.Get() == nullptr)
	{
		TexCoordVertex BoundingBox[] =
		{
			TexCoordVertex(-1.0f,	-1.0f,	1.0f,	0.0f,	1.0f), //Bottom Left 
			TexCoordVertex(-1.0f,	1.0f,	1.0f,	0.0f,	0.0f), //Top Left
			TexCoordVertex(1.0f,	1.0f,	1.0f,	1.0f,	0.0f), //Top Right

			TexCoordVertex(-1.0f,	-1.0f,	1.0f,	0.0f,	1.0f), //Bottom Left 
			TexCoordVertex(1.0f,	1.0f,	1.0f,	1.0f,	0.0f), //Top Right
			TexCoordVertex(1.0f,	-1.0f,	1.0f,	1.0f,	1.0f), //Bottom Right

		};


		this->vertexBuffer.Initialize(this->device, BoundingBox, ARRAYSIZE(BoundingBox));
	}



	return true;
}




bool Raycasting::createRaycastingShaderResourceView()
{

	if (shaderResourceView == nullptr)
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC shader_resource_view_desc;
		ZeroMemory(&shader_resource_view_desc, sizeof(shader_resource_view_desc));

		shader_resource_view_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		shader_resource_view_desc.Texture2D.MipLevels = 1;
		shader_resource_view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	

		HRESULT hr = this->device->CreateShaderResourceView(
			this->getTexture(),
			&shader_resource_view_desc,
			shaderResourceView.GetAddressOf()
		);

		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create shader resource view");
			return false;
		}
	}


	return true;
}


bool Raycasting::initializeSamplerstate()
{
	if (samplerState.Get() == nullptr)
	{
		//Create sampler description for sampler state
		D3D11_SAMPLER_DESC sampDesc;
		ZeroMemory(&sampDesc, sizeof(sampDesc));
		sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		sampDesc.MinLOD = 0;
		sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
		HRESULT hr = this->device->CreateSamplerState(&sampDesc, this->samplerState.GetAddressOf()); //Create sampler state
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to create sampler state.");
			return false;
		}
	}


	return true;
}



bool Raycasting::initializeRasterizer()
{
	if (this->rasterizerstate.Get() == nullptr)
	{
		// Create Rasterizer state
		D3D11_RASTERIZER_DESC rasterizerDesc;
		ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE; // CULLING could be set to none
		rasterizerDesc.MultisampleEnable = true;
		rasterizerDesc.AntialiasedLineEnable = true;
		//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;

		HRESULT hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
			return false;
		}



		//Create the blend state
		D3D11_BLEND_DESC blendDesc;
		ZeroMemory(&blendDesc, sizeof(blendDesc));

		D3D11_RENDER_TARGET_BLEND_DESC rtbd;
		ZeroMemory(&rtbd, sizeof(rtbd));

		rtbd.BlendEnable = true;
		rtbd.SrcBlend = D3D11_BLEND::D3D11_BLEND_SRC_ALPHA;
		rtbd.DestBlend = D3D11_BLEND::D3D11_BLEND_INV_SRC_ALPHA;
		rtbd.BlendOp = D3D11_BLEND_OP::D3D11_BLEND_OP_ADD;

		rtbd.SrcBlendAlpha = D3D11_BLEND::D3D11_BLEND_ONE;
		rtbd.DestBlendAlpha = D3D11_BLEND::D3D11_BLEND_ZERO;

		rtbd.BlendOpAlpha = D3D11_BLEND_OP::D3D11_BLEND_OP_ADD;
		rtbd.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE::D3D11_COLOR_WRITE_ENABLE_ALL;

		blendDesc.RenderTarget[0] = rtbd;

		hr = this->device->CreateBlendState(&blendDesc, this->blendState.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to create blend state.");
			return false;
		}

	}

	return true;
}

void Raycasting::setShaders()
{

	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout

	// set the primitive topology
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);		

	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);
	UINT offset = 0;

	// set Vertex buffer
	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); 
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	this->deviceContext->PSSetShaderResources(0, 1, this->shaderResourceView.GetAddressOf());
	this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF); 
	this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBuffer.GetAddressOf());

}


void Raycasting::draw()
{
	this->initializeRasterizer();
	this->initializeSamplerstate();
	this->createRaycastingShaderResourceView();

	this->initializeShaders();
	this->initializeScene();
	this->updateconstantBuffer();

	this->setShaders();
	this->deviceContext->Draw(6, 0);
}


template <typename Observable1, typename Observable2>
__global__ void CudaTerrainRenderer_extra_fluctuation_raycasting
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t t_isosurface,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions timeSpaceOptions,
	RaycastingOptions raycastingOptions
)
{

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox_spacetime.m_width;
		pixel.x = index - pixel.y * d_boundingBox_spacetime.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox_spacetime.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox_spacetime, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox_spacetime);



		//  if inside the bounding box
		if (NearFar.y != -1)
		{
			Observable1 observable1;
			Observable2 observable2;

			float3 rayDir = normalize(pixelPos - d_boundingBox_spacetime.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox_spacetime.m_eyePos + d_boundingBox_spacetime.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);
				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox_spacetime.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex(position, d_boundingBox_spacetime.m_dimensions, d_boundingBox_spacetime.gridSize);
				float3 relativePosIso = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
				// 0.5 is the offset for the texture coordinate
				float3 texPos = { relativePos.x,(float)timeSpaceOptions.wallNoramlPos + 0.5f , relativePos.z };


				float height = observable2.ValueAtXYZ_Tex(heightField, texPos);

				if (timeSpaceOptions.usingAbsolute)
				{
					height = abs(height) * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.m_dimensions.y / d_boundingBox_spacetime.gridSize.y;

				}
				else
				{
					height = height * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.m_dimensions.y / d_boundingBox_spacetime.gridSize.y;
				}

				int current = timeSpaceOptions.currentTime;
				bool skip = false;
				float sliderPos = (position.x) / (d_boundingBox_spacetime.m_dimensions.x / d_boundingBox_spacetime.gridSize.x);

				switch (timeSpaceOptions.sliderBackground)
				{
				case (SliderBackground::SliderBackground::BACKWARD):
				{
					if (sliderPos < current)
						skip = true;
					break;
				}
				case (SliderBackground::SliderBackground::FORWARD):
				{
					if (sliderPos > current)
						skip = true;
					break;
				}
				case (SliderBackground::SliderBackground::BAND):
				{
					if (sliderPos > current + timeSpaceOptions.bandSize || sliderPos < current - timeSpaceOptions.bandSize)
						skip = true;
					break;
				}
				}

				float value = observable1.ValueAtXYZ_Tex(t_isosurface, relativePosIso);

				//value = gaussianFilter(timeSpaceOptions.filterSize, t_isosurface, relativePosIso).x;


				if (value > raycastingOptions.isoValue_0)
				{

					//position = binarySearch<Observable>(observable, field1, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, dir * t, raycastingOptions.isoValue_0, raycastingOptions.tolerance_0, 200);
					relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

					// calculates gradient
					float3 gradient = observable2.GradientAtXYZ_Tex(t_isosurface, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox_spacetime.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_0);
					float3 rgb = raycastingColor * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox_spacetime.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}



				// Heightfield
				if (position.y - height < timeSpaceOptions.hegiht_tolerance && height - position.y < timeSpaceOptions.hegiht_tolerance && !skip)
				{

					float3 rgb = colorCode(timeSpaceOptions.minColor, timeSpaceOptions.maxColor, observable1.ValueAtXYZ_Tex(heightField, texPos), timeSpaceOptions.min_val, timeSpaceOptions.max_val);


					if (timeSpaceOptions.shading)
					{
						float3 gradient = observable2.GradientAtXYZ_Tex_Absolute(heightField, texPos, d_boundingBox_spacetime.m_dimensions, d_boundingBox_spacetime.gridSize);
						//gradient = normalize(make_float3(gradient.x, 0, gradient.z));
						gradient = normalize(make_float3(timeSpaceOptions.height_scale* gradient.x, -1, timeSpaceOptions.height_scale*gradient.z));
						// shading (no ambient)
						float diffuse = max(dot(gradient, viewDir), 0.0f);
						rgb = rgb * diffuse;
					}

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}
			}
		}
	}
}


template <typename Observable1, typename Observable2>
__global__ void CudaTerrainRenderer_extra_fluctuation
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions timeSpaceOptions
)
{

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox_spacetime.m_width;
		pixel.x = index - pixel.y * d_boundingBox_spacetime.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox_spacetime.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox_spacetime, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox_spacetime);



		//  if inside the bounding box
		if (NearFar.y != -1)
		{
			Observable1 observable1;
			Observable2 observable2;

			float3 rayDir = normalize(pixelPos - d_boundingBox_spacetime.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox_spacetime.m_eyePos + d_boundingBox_spacetime.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox_spacetime.m_dimensions / 2.0;
				float3 position_shifted = position;

				if (timeSpaceOptions.shiftProjection)
				{
					position_shifted.x += timeSpaceOptions.shiftProjectionPlane;
				}

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex(position_shifted, d_boundingBox_spacetime.m_dimensions, d_boundingBox_spacetime.gridSize);

				// 0.5 is the offset for the texture coordinate
				float3 texPos = {relativePos.x,(float)timeSpaceOptions.wallNoramlPos + 0.5f , relativePos.z};

				


				float height = 0;

				if (timeSpaceOptions.gaussianFilterHeight)
				{
					height = filterGaussian2D(timeSpaceOptions.filterSizeHeight, timeSpaceOptions.stdHeight, heightField, 2, texPos).y;
				}
				else
				{
					height = observable2.ValueAtXYZ_Tex(heightField, texPos);
					//height = cubicTex3DSimple(heightField, texPos).y;
				}


				if (timeSpaceOptions.usingAbsolute)
				{
					height = (abs(height) * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.m_dimensions.y/ d_boundingBox_spacetime.gridSize.y);

				}
				else
				{
					height = (height * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.m_dimensions.y / d_boundingBox_spacetime.gridSize.y);
				}

				int current = timeSpaceOptions.currentTime;
				bool skip = false;
				float sliderPos =  (position.x ) / (d_boundingBox_spacetime.m_dimensions.x / d_boundingBox_spacetime.gridSize.x);

				switch (timeSpaceOptions.sliderBackground)
				{
				case (SliderBackground::SliderBackground::BACKWARD):
				{
					if (sliderPos < current)
						skip = true;
					break;
				}
				case (SliderBackground::SliderBackground::FORWARD):
				{
					if (sliderPos > current)
						skip = true;
					break;
				}
				case (SliderBackground::SliderBackground::BAND):
				{
					if (sliderPos > current + timeSpaceOptions.bandSize || sliderPos < current - timeSpaceOptions.bandSize)
						skip = true;
					break;
				}
				}

				// Heightfield
				if ( position.y - height < timeSpaceOptions.hegiht_tolerance && height - position.y < timeSpaceOptions.hegiht_tolerance && !skip)
				{
					float value = 0;
					if (timeSpaceOptions.gaussianFilter)
					{
						value = observable1.ValueAtXYZ_Tex(heightField, texPos) - filterGaussian2D(timeSpaceOptions.filterSize, timeSpaceOptions.std, heightField, 2, texPos).x;
					}
					else
					{
						value = observable1.ValueAtXYZ_Tex(heightField, texPos);
					}
					//float3 rgb = colorCode(timeSpaceOptions.minColor, timeSpaceOptions.maxColor, observable1.ValueAtXYZ_Tex(heightField, texPos), timeSpaceOptions.min_val, timeSpaceOptions.max_val);
					float3 rgb = colorCode(timeSpaceOptions.minColor, timeSpaceOptions.maxColor, value, timeSpaceOptions.min_val, timeSpaceOptions.max_val);
					rgb = rgb * timeSpaceOptions.brightness;

					if (timeSpaceOptions.shading)
					{
						float3 gradient = observable2.GradientAtXYZ_Tex_Absolute(heightField, texPos, d_boundingBox_spacetime.m_dimensions, d_boundingBox_spacetime.gridSize);
						//gradient = normalize(make_float3(gradient.x, 0, gradient.z));
						gradient = normalize(make_float3(gradient.x * timeSpaceOptions.height_scale, -1, gradient.z * timeSpaceOptions.height_scale));
						// shading (no ambient)
						float diffuse = max(dot(gradient, viewDir), 0.0f);
						rgb = rgb * diffuse;
					}
					//rgb = rgb ^ timeSpaceOptions.brightness;
					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}
			}
		}
	}
}





template <>
__global__ void CudaCrossSectionRenderer<CrossSectionOptionsMode::SpanMode::TIME>
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	cudaTextureObject_t t_gradient,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SolverOptions solverOptions,
	CrossSectionOptions crossSectionOptions
)
{

	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < rays)
	{

		// 3D texture Size


		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;


				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = (position / d_boundingBox.m_dimensions);


				int dt = solverOptions.lastIdx + 1 - solverOptions.firstIdx;

				float positionOfSlice = (float)crossSectionOptions.wallNormalPos /(float)solverOptions.gridSize[1];
				float time = (float)crossSectionOptions.slice / (float)dt;


				// check if we have a hit 
				if (relativePos.y - positionOfSlice > 0 && relativePos.y - positionOfSlice < 0.001)
				{
					float value = tex3D<float4>(field1, relativePos.z,relativePos.x, time).x;
		
					float4 gradient = tex3D<float4>(t_gradient, relativePos.z, relativePos.x, time);

					float gradient_magnitude = dot(gradient, gradient);
					gradient_magnitude = gradient.x;

					float3 rgb_min =
					{
						crossSectionOptions.minColor[0],
						crossSectionOptions.minColor[1],
						crossSectionOptions.minColor[2],
					};

					float3 rgb_max =
					{
						crossSectionOptions.maxColor[0],
						crossSectionOptions.maxColor[1],
						crossSectionOptions.maxColor[2],
					};

					float3 rgb = { 0,0,0 };
					float y_saturated = 0.0f;

					if (value < 0)
					{
						float3 rgb_min_complement = make_float3(1, 1, 1) - rgb_min;
						y_saturated = saturate(abs(value / crossSectionOptions.min_val));
						rgb = rgb_min_complement * (1 - y_saturated) + rgb_min;
					}
					else
					{
						float3 rgb_max_complement = make_float3(1, 1, 1) - rgb_max;
						y_saturated = saturate(value / crossSectionOptions.max_val);
						rgb = rgb_max_complement * (1 - y_saturated) + rgb_max;
					}


					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel

					if (crossSectionOptions.filterMinMax)
					{
						if (fabsf(gradient_magnitude) > crossSectionOptions.min_max_threshold)
						{
							rgba = { 0,0,0,depth };
						}
					}

					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;
				}


			}



		}

	}
}


template <>
__global__ void CudaCrossSectionRenderer<CrossSectionOptionsMode::SpanMode::WALL_NORMAL>
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	cudaTextureObject_t gradient,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SolverOptions solverOptions,
	CrossSectionOptions crossSectionOptions
)
{

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				float positionOfSlice = (float)crossSectionOptions.slice / (float)d_boundingBox.gridSize.y;


				// check if we have a hit 
				if (position.y - positionOfSlice > 0 && position.y - positionOfSlice < 0.001)
				{
				float value = tex3D<float4>(field1, relativePos.x, relativePos.y, relativePos.z).x;

				float3 rgb_min =
				{
					crossSectionOptions.minColor[0],
					crossSectionOptions.minColor[1],
					crossSectionOptions.minColor[2],
				};

				float3 rgb_max =
				{
					crossSectionOptions.maxColor[0],
					crossSectionOptions.maxColor[1],
					crossSectionOptions.maxColor[2],
				};

				float3 rgb = { 0,0,0 };
				float y_saturated = 0.0f;

				if (value < 0)
				{
					float3 rgb_min_complement = make_float3(1, 1, 1) - rgb_min;
					y_saturated = saturate(abs(value / crossSectionOptions.min_val));
					rgb = rgb_min_complement * (1 - y_saturated) + rgb_min;
				}
				else
				{
					float3 rgb_max_complement = make_float3(1, 1, 1) - rgb_max;
					y_saturated = saturate(value / crossSectionOptions.max_val);
					rgb = rgb_max_complement * (1 - y_saturated) + rgb_max;
				}


				// vector from eye to isosurface
				float3 position_viewCoordinate = position - eyePos;

				// calculates the z-value
				float z_dist = abs(dot(viewDir, position_viewCoordinate));

				// calculate non-linear depth between 0 to 1
				float depth = (f) / (f - n);
				depth += (-1.0f / z_dist) * (f * n) / (f - n);

				float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

				// write back color and depth into the texture (surface)
				// stride size of 4 * floats for each texel
				surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
				break;
			}


			}



		}

	}
}



__global__ void CudaFilterExtremumX
(
	cudaSurfaceObject_t filtered,
	cudaTextureObject_t unfiltered,
	int2 dimension,
	float threshold,
	int z
)
{
	int index = CUDA_INDEX;
	
	if (index < dimension.x * dimension.y)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = IndexToPixel(index, dimension);


		float2 gradient = GradientXY_Tex3D_X(unfiltered, make_int3(pixel.x, pixel.y, z), dimension);
		float4 gradient_float4 = make_float4(gradient.x, gradient.y, 0.0f, 0.0f);

		surf3Dwrite(gradient_float4, filtered, 4 * sizeof(float) * pixel.x, pixel.y, z);


	}
}



template <typename Observable>
__global__ void CudaIsoSurfacRendererSpaceTime
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays, float isoValue,
	float samplingRate,
	float IsosurfaceTolerance
)
{

	Observable observable;

	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = (position / d_boundingBox.m_dimensions);


				// check if we have a hit 
				if (observable.ValueAtXYZ_Tex(field1, relativePos) - isoValue > 0)
				{

					position = binarySearch<Observable>(observable, field1, position, d_boundingBox.m_dimensions , d_boundingBox.gridSize, rayDir * t, isoValue, IsosurfaceTolerance, 50);
					relativePos = (position / d_boundingBox.m_dimensions);

					// calculates gradient
					float3 gradient = observable.GradientAtXYZ_Tex(field1, relativePos, d_boundingBox.m_dimensions,d_boundingBox.gridSize);

					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
					float3 rgb = make_float3(1,1,1) *diffuse;


					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;
				}


			}

		}

	}


}




__global__ void CudaTerrainRenderer_Marching_extra
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 t_range = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (t_range.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = t_range.x;
			
			float2 cellSize_XZ = make_float2(d_boundingBox.m_dimensions.x, d_boundingBox.m_dimensions.z) /
				make_int2(dispersionOptions.gridSize_2D[0], dispersionOptions.gridSize_2D[1]);
			
			// While the ray is within the Bounding Box
			while (t < t_range.y)
			{
				
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex
				(
					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
					make_float3(d_boundingBox.m_dimensions.x, d_boundingBox.m_dimensions.z, static_cast<float>(traceTime)),
					make_int3(dispersionOptions.gridSize_2D[0], dispersionOptions.gridSize_2D[1], static_cast<float> (traceTime))
				);

				// fetch texel from the GPU memory
				float4 hightFieldVal = ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

					
					float3 gradient = { hightFieldVal.z,-1,hightFieldVal.w };
					
					
					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);


					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);

					float3 rgb = rgb_min;

					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}
				if (!dispersionOptions.marching)
				{
					t = t + samplingRate;
				}
				else
				{
					t = t + fmin(samplingRate, findExitPoint(make_float2(position.x, position.z), make_float2(rayDir.x, rayDir.z), cellSize_XZ));

				}

			}

		}

	}
}








__global__ void CudaTerrainRenderer_Marching_extra_FSLE
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 t_range = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (t_range.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = t_range.x;


			// While the ray is within the Bounding Box
			while (t < t_range.y)
			{

				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex
				(
					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
					make_float3(d_boundingBox.m_dimensions.x, d_boundingBox.m_dimensions.z, d_boundingBox.gridSize.z),
					d_boundingBox.gridSize
				);
				
				// fetch texel from the GPU memory
				float4 hightFieldVal = cubicTex3DSimple(heightField, relativePos);


				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

					float3 gradient = { hightFieldVal.y,-1,hightFieldVal.z };
					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);


					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);
					float fsle = 0.0f;

					if (dispersionOptions.forward)
						fsle = ValueAtXYZ_float4(extraField, relativePos).x;
					else
						fsle = ValueAtXYZ_float4(extraField, relativePos).y;

					float3 rgb = { 0,0,0 };

					float extractedVal = saturate((fsle - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val)) * 2;
					if (extractedVal > 1)
					{
						rgb = saturateRGB(rgb_max, extractedVal - 1);
					}
					else
					{
						rgb = saturateRGB(rgb_min, 1-extractedVal);
					}


					//float3 rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);



					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}
				if (!dispersionOptions.marching)
				{
					t = t + samplingRate;
				}
				//else
				//{
				//	t = t + fmin(samplingRate, findExitPoint(make_float2(position.x, position.z), make_float2(rayDir.x, rayDir.z), cellSize_XZ));

				//}

			}

		}

	}
}



__global__ void CudaTerrainRenderer_Marching_extra_FTLE_Color
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox.m_width;
		pixel.x = index - pixel.y * d_boundingBox.m_width;

		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 t_range = findIntersections(pixelPos, d_boundingBox);


		// if inside the bounding box
		if (t_range.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = t_range.x;


			// While the ray is within the Bounding Box
			while (t < t_range.y)
			{

				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex
				(
					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
					make_float3(d_boundingBox.m_dimensions.x, d_boundingBox.m_dimensions.z, d_boundingBox.gridSize.z),
					d_boundingBox.gridSize
				);

				// fetch texel from the GPU memory
				float4 ftle = ValueAtXYZ_float4(extraField, relativePos);
				ftle.x *= dispersionOptions.scale;


				// check if we have a hit 
				if (position.y - ftle.x > 0 && position.y - ftle.x < dispersionOptions.hegiht_tolerance)
				{
					float3 gradient = normalize(GradientAtXYZ_Tex_X_Height(extraField, relativePos));
					gradient = normalize(make_float3(gradient.x, -1.0f, gradient.y));

					// shading (no ambient)
					float diffuse = max(dot(gradient, viewDir), 0.0f);
					float height = ValueAtXYZ_float4(extraField, relativePos).x;


					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);
					float3 rgb = rgb_min;


					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);



					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					break;
				}
				if (!dispersionOptions.marching)
				{
					t = t + samplingRate;
				}

			}

		}

	}
}




__device__ float3 binarySearch_tex1D
(
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	int3& gridSize,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
)
{
	float3 position = _position;
	float3 relative_position = world2Tex(position, gridDiameter, gridSize);
	float3 samplingStep = _samplingStep * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;

	while (fabsf(tex3D<float>(field, relative_position.x, relative_position.y, relative_position.z) - value) > tolerance&& counter < maxIteration)
	{

		if (tex3D<float>(field, relative_position.x, relative_position.y, relative_position.z) - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			position = position - samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			position = position + samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 1;

		}
		counter++;

	}

	return position;

};



void Raycasting::generateMipmapL1()
{
	int3 mipmapGridSize = Array2Int3(solverOptions->gridSize);
	mipmapGridSize = mipmapGridSize / 2;

	s_mipmapped.setInputArray(a_mipmap_L1.getArrayRef());
	s_mipmapped.initializeSurface();

	// Calculates the block and grid sizes

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(mipmapGridSize.x * mipmapGridSize.y);

	for (int z = 0; z < mipmapGridSize.z; z++)
	{
		mipmapped << < blocks, thread >> > (volumeTexture_0.getTexture(), s_mipmapped.getSurfaceObject(), mipmapGridSize, z);
	}

	s_mipmapped.destroySurface();

}

void Raycasting::initializeMipmapL1()
{
	a_mipmap_L1.release();
	int3 mipmapGridSize = Array2Int3(solverOptions->gridSize);
	mipmapGridSize = mipmapGridSize / 2;
	a_mipmap_L1.initialize(mipmapGridSize.x, mipmapGridSize.y, mipmapGridSize.z);
}

void Raycasting::initializeMipmapL2()
{
	a_mipmap_L2.release();
	int3 mipmapGridSize = Array2Int3(solverOptions->gridSize);
	mipmapGridSize = mipmapGridSize / 4;
	a_mipmap_L2.initialize(mipmapGridSize.x, mipmapGridSize.y, mipmapGridSize.z);
}
void Raycasting::generateMipmapL2()
{
	int3 mipmapGridSize = Array2Int3(solverOptions->gridSize);
	mipmapGridSize = mipmapGridSize / 4;
	s_mipmapped.setInputArray(a_mipmap_L2.getArrayRef());
	s_mipmapped.initializeSurface();

	// Calculates the block and grid sizes

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(mipmapGridSize.x * mipmapGridSize.y);

	for (int z = 0; z < mipmapGridSize.z; z++)
	{
		mipmapped << < blocks, thread >> > (volumeTexture_L1.getTexture(), s_mipmapped.getSurfaceObject(), mipmapGridSize, z);
	}

	s_mipmapped.destroySurface();
}



