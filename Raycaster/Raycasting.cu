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
	int3 & gridSize,
	VolumeTexture3D & volumeTexture,
	Volume_IO_Z_Major & volume_IO,
	const int & idx,
	bool isCompressed,
	const int & memberIdx,
	cudaTextureAddressMode addressModeX,
	cudaTextureAddressMode addressModeY,
	cudaTextureAddressMode addressModeZ)
{

	Timer timer;
	// Read current volume
	if (isCompressed)
	{
		volume_IO.readVolume_Compressed(idx, gridSize,memberIdx);
		float * h_VelocityField = this->volume_IO.getField_float_GPU();// set the pointer to the volume texture
		volumeTexture.setField(h_VelocityField);
		// initialize the volume texture
		TIMELAPSE(volumeTexture.initialize_devicePointer(gridSize, false, addressModeX, addressModeY, addressModeZ);, "Initialize Texture including DDCopy");
		cudaFree(h_VelocityField);
	}
	else {
		// Read current volume
		volume_IO.readVolume(idx, memberIdx);
		float * h_VelocityField = volume_IO.getField_float();
		// set the pointer to the volume texture
		volumeTexture.setField(h_VelocityField);
		// initialize the volume texture
		volumeTexture.initialize(gridSize, false, addressModeX, addressModeY, addressModeZ);
		// release host memory
		volume_IO.release();
	}
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
	this->volume_IO.readVolume_Compressed(idx, Array2Int3(gridSize0));
	float * h_VelocityField = this->volume_IO.getField_float_GPU();
	// set the pointer to the volume texture
	volumeTexture_0.setField(h_VelocityField);
	// initialize the volume texture
	TIMELAPSE(volumeTexture_0.initialize_devicePointer(Array2Int3(gridSize0), false, addressModeX, addressModeY, addressModeZ); , "Initialize Texture including DDCopy");
	// Free host memory
	cudaFree(h_VelocityField);


	//Same procedure for the secondary dataset
	this->volume_IO.readVolume_Compressed(idx, Array2Int3(gridSize1));
	h_VelocityField = this->volume_IO.getField_float_GPU();
	volumeTexture_1.setField(h_VelocityField);
	TIMELAPSE(volumeTexture_1.initialize_devicePointer(Array2Int3(gridSize1), false, addressModeX, addressModeY, addressModeZ);, "Initialize Texture including DDCopy");
	cudaFree(h_VelocityField);
}



__host__ bool Raycasting::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z 
)
{
	if (!this->initialized)
	{
		if (!this->initializeRaycastingTexture())
			return false;
		if (!this->initializeBoundingBox())
			return false;
		this->rays = (*this->width) * (*this->height);

		this->volume_IO.Initialize(&this->fieldOptions[raycastingOptions->raycastingField_0]);
		int3 gridSize = Array2Int3(solverOptions->gridSize);
		switch (raycastingOptions->raycastingMode)
		{
		case RaycastingMode::Mode::SINGLE:
		case RaycastingMode::Mode::DVR:
		case RaycastingMode::Mode::PLANAR:
		case RaycastingMode::Mode::PROJECTION_BACKWARD:
		case RaycastingMode::Mode::PROJECTION_FORWARD:
		case RaycastingMode::Mode::PROJECTION_AVERAGE:
		case RaycastingMode::Mode::PROJECTION_LENGTH:
			loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
			//volume_IO.release();
			break;
		case RaycastingMode::Mode::DOUBLE:
		case RaycastingMode::Mode::PLANAR_DOUBLE:
		case RaycastingMode::Mode::DOUBLE_SEPARATE:
		case RaycastingMode::Mode::DVR_DOUBLE:
		case RaycastingMode::Mode::DOUBLE_ADVANCED:
		case RaycastingMode::Mode::DOUBLE_TRANSPARENCY:
			this->volume_IO.Initialize(&this->fieldOptions[raycastingOptions->raycastingField_0]);
			loadTexture(gridSize, this->volumeTexture_0, volume_IO, fieldOptions[raycastingOptions->raycastingField_0].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
			volume_IO.release();
			this->volume_IO.Initialize(&this->fieldOptions[raycastingOptions->raycastingField_1]);
			loadTexture(gridSize, this->volumeTexture_1, volume_IO, fieldOptions[raycastingOptions->raycastingField_1].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
			volume_IO.release();
			break;
		}

		this->initialized = true;
	}
	return true;

}



__host__ bool Raycasting::initialize_Double
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	if (!this->initializeRaycastingTexture())
		return false;


	if (!this->initializeBoundingBox())
		return false;

	this->rays = (*this->width) * (*this->height);

	this->volume_IO.Initialize(this->fieldOptions);

	int3 gridSize = Array2Int3(solverOptions->gridSize);

	loadTexture(gridSize, this->volumeTexture_0, volume_IO, fieldOptions[raycastingOptions->raycastingField_0].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
	loadTexture(gridSize, this->volumeTexture_1, volume_IO, fieldOptions[raycastingOptions->raycastingField_1].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);


	this->initialized = true;

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
	this->volume_IO.Initialize(this->fieldOptions);
	int3 gridSize = Array2Int3(solverOptions->gridSize);


		
	loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx,fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);


	initializeMipmapL1();
	generateMipmapL1();
	volumeTexture_L1.setArray(a_mipmap_L1.getArrayRef());
	volumeTexture_L1.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	initializeMipmapL2();
	generateMipmapL2();
	volumeTexture_L2.setArray(a_mipmap_L2.getArrayRef());
	volumeTexture_L2.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);


	this->initialized = true;

	return true;

}




__host__ bool Raycasting::release()
{
	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
	this->volumeTexture_L1.release();
	this->volumeTexture_L2.release();
	this->a_mipmap_L1.release();
	this->a_mipmap_L2.release();
	this->volume_IO.release();
	this->raycastingSurface.destroySurface();

	this->raycastingTexture.Reset();
	this->renderTargetView.Reset();
	this->rasterizerstate.Reset();
	this->samplerState.Reset();
	this->shaderResourceView.Reset();
	this->blendState.Reset();

	this->initialized = false;
	return true;
}


void Raycasting::updateFile
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
	int3 gridSize = Array2Int3(solverOptions->gridSize);

	switch (raycastingOptions->raycastingMode)
	{
	case RaycastingMode::Mode::SINGLE:
	case RaycastingMode::Mode::PLANAR:
	case RaycastingMode::Mode::PROJECTION_BACKWARD:
	case RaycastingMode::Mode::PROJECTION_FORWARD:
	case RaycastingMode::Mode::PROJECTION_AVERAGE:
	case RaycastingMode::Mode::PROJECTION_LENGTH:
	case RaycastingMode::Mode::DVR:

		loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx, fieldOptions[raycastingOptions->raycastingField_0].isCompressed, fieldOptions[raycastingOptions->raycastingField_0].firstMemberIdx, addressMode_X, addressMode_Y, addressMode_Z);
		break;

	case RaycastingMode::Mode::DOUBLE:
	case RaycastingMode::Mode::DVR_DOUBLE:
	case RaycastingMode::Mode::PLANAR_DOUBLE:
	case RaycastingMode::Mode::DOUBLE_SEPARATE:
	case RaycastingMode::Mode::DOUBLE_ADVANCED:
	case RaycastingMode::Mode::DOUBLE_TRANSPARENCY:
		loadTexture(gridSize, this->volumeTexture_0, volume_IO, fieldOptions[raycastingOptions->raycastingField_0].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
		volume_IO.release();
		this->volume_IO.Initialize(&this->fieldOptions[raycastingOptions->raycastingField_1]);
		loadTexture(gridSize, this->volumeTexture_1, volume_IO, fieldOptions[raycastingOptions->raycastingField_1].currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
		break;
	}



}



void Raycasting::updateFile_Double
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
	int3 gridSize = Array2Int3(solverOptions->gridSize);

	loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx,fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
	loadTexture(gridSize, this->volumeTexture_1, volume_IO, solverOptions->currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);
	
}




void Raycasting::updateFile_MultiScale
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)
{
	this->volumeTexture_0.release();
	this->volumeTexture_L1.release();
	this->volumeTexture_L2.release();

	int3 gridSize = Array2Int3(solverOptions->gridSize);

	loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx, fieldOptions->isCompressed, addressMode_X, addressMode_Y, addressMode_Z);

	generateMipmapL1();
	volumeTexture_L1.setArray(a_mipmap_L1.getArrayRef());
	volumeTexture_L1.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

	generateMipmapL2();
	volumeTexture_L2.setArray(a_mipmap_L2.getArrayRef());
	volumeTexture_L2.initialize_array(false, addressMode_X, addressMode_Y, addressMode_Z);

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
				*this->raycastingOptions,
				*this->renderingOptions
				);

		break;
	case RaycastingMode::Mode::DVR:

		CudaDVR << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);

		break;

	case RaycastingMode::Mode::DVR_DOUBLE:

		if(raycastingOptions->within)
				CudaDVR_Double_Within << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		else
			CudaDVR_Double << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);

		break;

	case RaycastingMode::Mode::SINGLE_COLORCODED:

		CudaIsoSurfacRenderer_Single_ColorCoded << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);

		break;

	case RaycastingMode::Mode::DOUBLE:

		CudaIsoSurfacRenderer_Double_Modes << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);

		break;

		//CudaIsoSurfacRenderer_Double << < blocks, thread >> >
		//	(
		//		this->raycastingSurface.getSurfaceObject(),
		//		this->volumeTexture_0.getTexture(),
		//		this->volumeTexture_1.getTexture(),
		//		int(this->rays),
		//		*this->raycastingOptions,
		//		*this->renderingOptions
		//		);
		//break;

	case RaycastingMode::Mode::DOUBLE_SEPARATE:


		CudaIsoSurfacRenderer_Double_Separate << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::MULTISCALE:

		CudaIsoSurfacRenderer_Multiscale << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_L1.getTexture(),
				this->volumeTexture_L2.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;


	case RaycastingMode::Mode::MULTISCALE_TEMP:

		CudaIsoSurfacRenderer_Double_Transparency_noglass_multiLevel << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_L1.getTexture(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::MULTISCALE_DEFECT:

		CudaIsoSurfacRenderer_Multiscale_Defect << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_L1.getTexture(),
				this->volumeTexture_L2.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::PLANAR:

		CudaIsoSurfacRenderer_Planar << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::PLANAR_DOUBLE:

		CudaIsoSurfacRenderer_Planar_Double << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;
	case RaycastingMode::Mode::PROJECTION_FORWARD:

		CudaIsoSurfacRenderer_Projection_Forward << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::PROJECTION_BACKWARD:

		CudaIsoSurfacRenderer_Projection_Backward << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::PROJECTION_AVERAGE:

		CudaIsoSurfacRenderer_Projection_Average << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;	
	
	case RaycastingMode::Mode::PROJECTION_LENGTH:

		CudaIsoSurfacRenderer_Projection_Length << <blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::DOUBLE_ADVANCED:
		CudaIsoSurfacRenderer_Double_Advanced << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;

	case RaycastingMode::Mode::DOUBLE_TRANSPARENCY:
		CudaIsoSurfacRenderer_Double_Transparency_noglass << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions
				);
		break;
		
	}

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
	

	h_boundingBox->FOV = static_cast<float>(((double)renderingOptions->FOV_deg / 360.0)* (double)XM_2PI);
	h_boundingBox->distImagePlane = this->distImagePlane;
	h_boundingBox->gridDiameter = ArrayToFloat3(solverOptions->gridDiameter);
	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));

	delete h_boundingBox;
	

	return true;
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

				float value = observable.ValueAtXYZ_Tex(field1, relativePos);
				float2 tEnterExit = { t,findExitPoint3D(position, dir, cellSize) };

				//float4 factors = getFactors<float>(values, position, dir);
				// check if we have a hit 
				if (value > raycastingOptions.isoValue_0)
				{

					//position = binarySearch<Observable>(observable, field1, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, dir * t, raycastingOptions.isoValue_0, raycastingOptions.tolerance_0, 200);
					relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

					// calculates gradient
					float3 gradient = observable.GradientAtXYZ_Tex(field1, relativePos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);


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



__global__ void CudaTerrainRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	RenderingOptions renderingOptions,
	int traceTime
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);
		float3 viewDir = dir;

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += gridDiameter / 2.0;


				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = 
				{ 
					position.x / d_boundingBox.gridDiameter.x,
					position.z / d_boundingBox.gridDiameter.z,
					static_cast<float> (dispersionOptions.timestep) / static_cast<float> (traceTime)
				};
				
				// fetch texels from the GPU memory
				float4 hightFieldVal = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 &&  position.y - hightFieldVal.x < 0.01 )
				{

					hightFieldVal = make_float4(0, 0, 0, 0); //ValueAtXYZ_float4(heightField, relativePos);

					float3 gradient = { -hightFieldVal.y,-1,-hightFieldVal.z };


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), rayDir), 0.0f);

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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;
				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos =
				{
					position.x / d_boundingBox.gridDiameter.x,
					position.z / d_boundingBox.gridDiameter.z,
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos =
				{
					position.z / d_boundingBox.gridDiameter.z,
					position.x / d_boundingBox.gridDiameter.x,
					static_cast<float> (dispersionOptions.timestep) / static_cast<float> (traceTime)
				};

				// fetch texels from the GPU memory
				float4 hightFieldVal = { 0,0,0,0 };// ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

		
					hightFieldVal = { 0,0,0,0 }; //ValueAtXYZ_float4(heightField, relativePos);
					
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

							float4 value = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(extraField, relativePos);
							extractedVal = value.y;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::V_Y:
						{

							extractedVal = 0;//ValueAtXYZ_float4(extraField, relativePos).z;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::V_Z:
						{

							extractedVal =0;//ValueAtXYZ_float4(extraField, relativePos).w;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DISTANCE:
						{
							float4 Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y,0.0));
							float4 Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y,0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { Vec_0.w,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DISTANCE_ZY:
						{
							float4 Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { 0.0f,Vec_0.x,Vec_1.x };

							Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { 0.0f,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionOptionsMode::dispersionColorCode::DEV_Z:
						{
							float4 Vec_1 = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float position_0 = Vec_1.x;

							Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, relativePos);

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
							float4 Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = make_float4(0, 0, 0, 0);// ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, relativePos);

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
							float4 Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = make_float4(0, 0, 0, 0);//ValueAtXYZ_float4(extraField, relativePos);

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
			this->raycastingTexture.Get(),
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
__global__ void CudaTerrainRenderer_extra_fluctuation
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions timeSpaceOptions,
	RenderingOptions renderingOptions
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
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox_spacetime.m_eyePos + d_boundingBox_spacetime.gridDiameter / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox_spacetime.gridDiameter / 2.0;
				float3 position_shifted = position;


				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex(position_shifted, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);

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
					height = (abs(height) * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.gridDiameter.y/ d_boundingBox_spacetime.gridSize.y);

				}
				else
				{
					height = (height * timeSpaceOptions.height_scale + timeSpaceOptions.offset + (float)timeSpaceOptions.wallNoramlPos * d_boundingBox_spacetime.gridDiameter.y / d_boundingBox_spacetime.gridSize.y);
				}

				int current = timeSpaceOptions.currentTime;
				bool skip = false;
				float sliderPos =  (position.x ) / (d_boundingBox_spacetime.gridDiameter.x / d_boundingBox_spacetime.gridSize.x);

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
						float3 gradient = observable2.GradientAtXYZ_Tex_Absolute(heightField, texPos, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);
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




__global__ void CudaTerrainRenderer_height_isoProjection
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions spaceTimeOptions,
	RenderingOptions renderingOptions
)
{

	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel;
		pixel.y = index / d_boundingBox_spacetime.m_width;
		pixel.x = index - pixel.y * d_boundingBox_spacetime.m_width;
		float3 eyePos = d_boundingBox_spacetime.m_eyePos + d_boundingBox_spacetime.gridDiameter / 2.0;
		// copy values from constant memory to local memory (which one is faster?)
		float3 viewDir = d_boundingBox_spacetime.m_viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox_spacetime, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox_spacetime);
		float4 rgba;
		float3 rgb;
		float3 position;
		float3 texPos;
		float depth = 1;
		float value = 0;
		bool hit = false;
		float t = NearFar.x;
		float3 rayDir = normalize(pixelPos - d_boundingBox_spacetime.m_eyePos);

		//  if inside the bounding box
		if (NearFar.y != -1)
		{


			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;
			bool hit = false;

			while (t < NearFar.y)
			{

				position = pixelPos + (rayDir * t);
				position = position + d_boundingBox_spacetime.gridDiameter / 2.0;
				texPos = world2Tex(position, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);
				float height = callerValueAtTex(spaceTimeOptions.heightMode, heightField, texPos);
				height *= spaceTimeOptions.height_scale;
				height+= (float)spaceTimeOptions.wallNoramlPos * d_boundingBox_spacetime.gridDiameter.y / d_boundingBox_spacetime.gridSize.y;
				if (position.y - height < spaceTimeOptions.hegiht_tolerance && height - position.y < spaceTimeOptions.hegiht_tolerance)
				{

					float3 rgb = colorCode(spaceTimeOptions.minColor, spaceTimeOptions.maxColor, height, spaceTimeOptions.min_val, spaceTimeOptions.max_val);
					rgb = rgb * spaceTimeOptions.brightness;

					if (spaceTimeOptions.shading)
					{
						float3 gradient = callerGradientAtTex(spaceTimeOptions.spaceTimeMode_0, heightField, texPos, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);
						gradient = normalize(make_float3(gradient.x * spaceTimeOptions.height_scale, -1, gradient.z * spaceTimeOptions.height_scale));
						float diffuse = max(dot(gradient, viewDir), 0.0f);
						rgb = rgb * diffuse;
					}


					// calculate non-linear depth between 0 to 1
					float depth = depthfinder(position, eyePos, viewDir, f, n);
					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					hit = true;
					break;
				}
				else
				{
					t = t + spaceTimeOptions.samplingRate_0;
				}
			}
			
			//// planar isosurface
			//if (spaceTimeOptions.additionalRaycasting)
			//{
			//	float t_x = d_boundingBox.gridDiameter.x / 2.0;
			//	t_x = t_x + ((float)spaceTimeOptions.timePosition * d_boundingBox.gridDiameter.x / (d_boundingBox.gridSize.x - 1));
			//	t_x = t_x - pixelPos.x;
			//	t_x = t_x / rayDir.x;

			//	if (t_x >= NearFar.x && t_x <= NearFar.y)
			//	{
			//		position = pixelPos + (rayDir * t_x);
			//		position = position + d_boundingBox_spacetime.gridDiameter / 2.0;
			//		depth = depthfinder(position, eyePos, viewDir, f, n);


			//		texPos = world2Tex(position, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);
			//		value = callerValueAtTex(spaceTimeOptions.heightMode,field,	make_float3(spaceTimeOptions.timePosition, texPos.y, texPos.z),d_boundingBox.gridDiameter,d_boundingBox.gridSize);
			//		rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, value, renderingOptions.minMeasure, renderingOptions.maxMeasure);
			//		rgba = { rgb.x , rgb.y, rgb.z, depth };
			//		surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
			//		hit = true;
			//	}
			//}

			//else
			//{
			//	// Projection of isosurfaces
			//	float t_x = -d_boundingBox_spacetime.gridDiameter.x / 2.0;
			//	t_x = t_x + ((float)spaceTimeOptions.timePosition * d_boundingBox_spacetime.gridDiameter.x / (d_boundingBox_spacetime.gridSize.x - 1));
			//	t_x = t_x - pixelPos.x;
			//	t_x = t_x / rayDir.x;

			//	if (t_x >= NearFar.x && t_x <= NearFar.y)
			//	{
			//		position = pixelPos + (rayDir * t_x);
			//		position = position + d_boundingBox_spacetime.gridDiameter / 2.0;



			//		texPos = world2Tex(position, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);

			//		int nStep_t = 0;
			//		float distance = 0;

			//		while (true)
			//		{
			//			value = callerValueAtTex(
			//				spaceTimeOptions.heightMode,
			//				heightField,
			//				make_float3(texPos.x + spaceTimeOptions.samplingRatio_t * nStep_t, texPos.y, texPos.z),
			//				d_boundingBox_spacetime.gridDiameter,
			//				d_boundingBox_spacetime.gridSize
			//			);

			//			if (value < spaceTimeOptions.isoValue)
			//			{
			//				break;
			//			}

			//			nStep_t++;

			//			if (texPos.x + spaceTimeOptions.samplingRatio_t * nStep_t > d_boundingBox_spacetime.gridSize.x)
			//			{
			//				break;
			//			}
			//		}

			//		distance = nStep_t * spaceTimeOptions.samplingRatio_t;
			//		rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);
			//		depth = depthfinder(position, eyePos, viewDir, f, n);

			//		rgba = { rgb.x , rgb.y, rgb.z, depth };
			//		surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
			//		hit = true;

			//	}
			//	
			//}
			//


			if (!hit)
			{
				// Space-time 
				float t_y = -pixelPos.y;
				t_y = t_y - d_boundingBox_spacetime.gridDiameter.y / 2.0;
				t_y = t_y + (float)spaceTimeOptions.wallNoramlPos * d_boundingBox_spacetime.gridDiameter.y / d_boundingBox_spacetime.gridSize.y;
				t_y = t_y / rayDir.y;

				if (t_y >= NearFar.x && t_y <= NearFar.y)
				{
					position = pixelPos + (rayDir * t_y);
					position = position + d_boundingBox_spacetime.gridDiameter / 2.0 ;

					if (spaceTimeOptions.shifSpaceTime)
					{
						position.x += spaceTimeOptions.projectionPlanePos;
						eyePos.x += spaceTimeOptions.projectionPlanePos;

					}

					float depth_new = depthfinder(position, eyePos, viewDir, f, n);

					if (depth_new < depth)
					{
						texPos = world2Tex(position, d_boundingBox_spacetime.gridDiameter, d_boundingBox_spacetime.gridSize);
						texPos = { texPos.x,(float)spaceTimeOptions.wallNoramlPos + 0.5f , texPos.z };

						value = callerValueAtTex(0, heightField,texPos);

						rgb = colorCode(spaceTimeOptions.minColor, spaceTimeOptions.maxColor, value, spaceTimeOptions.min_val, spaceTimeOptions.max_val);
						rgba = { rgb.x , rgb.y, rgb.z, depth_new };
						surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
					}

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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;


				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = (position / d_boundingBox.gridSize);


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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;



				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

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



//template <typename Observable>
//__global__ void CudaIsoSurfacRendererSpaceTime
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t field1,
//	int rays, float isoValue,
//	float samplingRate,
//	float IsosurfaceTolerance
//)
//{
//
//	Observable observable;
//
//	int index = blockIdx.x * blockDim.y * blockDim.x;
//	index += threadIdx.y * blockDim.x;
//	index += threadIdx.x;
//
//	if (index < rays)
//	{
//
//		// determine pixel position based on the index of the thread
//		int2 pixel;
//		pixel.y = index / d_boundingBox.m_width;
//		pixel.x = index - pixel.y * d_boundingBox.m_width;
//
//		// copy values from constant memory to local memory (which one is faster?)
//		float3 viewDir = d_boundingBox.m_viewDir;
//		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
//		float2 NearFar = findIntersections(pixelPos, d_boundingBox);
//
//
//		// if inside the bounding box
//		if (NearFar.y != -1)
//		{
//
//			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
//
//			// near and far plane
//			float n = 0.1f;
//			float f = 1000.0f;
//
//			// Add the offset to the eye position
//			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
//
//			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
//			{
//				// Position of the isosurface
//				float3 position = pixelPos + (rayDir * t);
//
//				// Adds an offset to position while the center of the grid is at gridDiamter/2
//				position += d_boundingBox.gridDiameter / 2.0;
//
//
//
//				//Relative position calculates the position of the point on the cuda texture
//				float3 relativePos = (position / d_boundingBox.gridDiameter);
//
//
//				// check if we have a hit 
//				if (observable.ValueAtXYZ_Tex(field1, relativePos) - isoValue > 0)
//				{
//
//					position = binarySearch<Observable>(observable, field1, position, d_boundingBox.gridDiameter , d_boundingBox.gridSize, rayDir * t, isoValue, IsosurfaceTolerance, 50);
//					relativePos = (position / d_boundingBox.gridDiameter);
//
//					// calculates gradient
//					float3 gradient = observable.GradientAtXYZ_Tex(field1, relativePos, d_boundingBox.gridDiameter,d_boundingBox.gridSize);
//
//					// shading (no ambient)
//					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
//					float3 rgb = make_float3(1,1,1) *diffuse;
//
//
//					// vector from eye to isosurface
//					float3 position_viewCoordinate = position - eyePos;
//
//					// calculates the z-value
//					float z_dist = abs(dot(viewDir, position_viewCoordinate));
//
//					// calculate non-linear depth between 0 to 1
//					float depth = (f) / (f - n);
//					depth += (-1.0f / z_dist) * (f * n) / (f - n);
//
//					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };
//
//					// write back color and depth into the texture (surface)
//					// stride size of 4 * floats for each texel
//					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
//					break;
//				}
//
//
//			}
//
//		}
//
//	}
//
//
//}




__global__ void CudaTerrainRenderer_Marching_extra
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	RenderingOptions renderingOptions,
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
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;


			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = t_range.x;
			
			float2 cellSize_XZ = make_float2(d_boundingBox.gridDiameter.x, d_boundingBox.gridDiameter.z) /
				make_int2(dispersionOptions.gridSize_2D[0], dispersionOptions.gridSize_2D[1]);
			
			// While the ray is within the Bounding Box
			while (t < t_range.y)
			{
				
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;

				//Relative position calculates the position of the point on the cuda texture
				float3 relativePos = world2Tex
				(
					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
					make_float3(d_boundingBox.gridDiameter.x, d_boundingBox.gridDiameter.z, static_cast<float>(traceTime)),
					make_int3(dispersionOptions.gridSize_2D[0], dispersionOptions.gridSize_2D[1], static_cast<float> (traceTime))
				);

				// fetch texel from the GPU memory
				//float4 hightFieldVal = callerValueAtTex(0,heightField, relativePos,gridDa);
				float4 heightFieldVal = make_float4(0, 0, 0, 0);
				// check if we have a hit 
				if (position.y - heightFieldVal.x > 0 && position.y - heightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

					
					float3 gradient = { heightFieldVal.z,-1,heightFieldVal.w };
					
					
					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);


					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);


					float3 rgb = make_float3(0, 0, 0);
						//colorCode(dispersionOptions.minColor, dispersionOptions.maxColor,,;

					rgb = rgb * diffuse;

					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = depthfinder(position, eyePos, viewDir, f, n);

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








//__global__ void CudaTerrainRenderer_Marching_extra_FSLE
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t heightField,
//	cudaTextureObject_t extraField,
//	int rays,
//	float samplingRate,
//	float IsosurfaceTolerance,
//	DispersionOptions dispersionOptions
//)
//{
//	int index = CUDA_INDEX;
//
//	if (index < rays)
//	{
//
//		// determine pixel position based on the index of the thread
//		int2 pixel;
//		pixel.y = index / d_boundingBox.m_width;
//		pixel.x = index - pixel.y * d_boundingBox.m_width;
//
//		// copy values from constant memory to local memory (which one is faster?)
//		float3 viewDir = d_boundingBox.m_viewDir;
//		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
//		float2 t_range = findIntersections(pixelPos, d_boundingBox);
//
//
//		// if inside the bounding box
//		if (t_range.y != -1)
//		{
//
//			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
//
//			// near and far plane
//			float n = 0.1f;
//			float f = 1000.0f;
//
//			// Add the offset to the eye position
//			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
//			float t = t_range.x;
//
//
//			// While the ray is within the Bounding Box
//			while (t < t_range.y)
//			{
//
//				// Position of the isosurface
//				float3 position = pixelPos + (rayDir * t);
//
//				// Adds an offset to position while the center of the grid is at gridDiamter/2
//				position += d_boundingBox.gridDiameter / 2.0;
//
//				//Relative position calculates the position of the point on the cuda texture
//				float3 relativePos = world2Tex
//				(
//					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
//					make_float3(d_boundingBox.gridDiameter.x, d_boundingBox.gridDiameter.z, d_boundingBox.gridSize.z),
//					d_boundingBox.gridSize
//				);
//				
//				// fetch texel from the GPU memory
//				float4 hightFieldVal = cubicTex3DSimple(heightField, relativePos);
//
//
//				// check if we have a hit 
//				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
//				{
//
//					float3 gradient = { hightFieldVal.y,-1,hightFieldVal.z };
//					// shading (no ambient)
//					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
//
//
//					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
//					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);
//					float fsle = 0.0f;
//
//					if (dispersionOptions.forward)
//						fsle = ValueAtXYZ_float4(extraField, relativePos).x;
//					else
//						fsle = ValueAtXYZ_float4(extraField, relativePos).y;
//
//					float3 rgb = { 0,0,0 };
//
//					float extractedVal = saturate((fsle - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val)) * 2;
//					if (extractedVal > 1)
//					{
//						rgb = saturateRGB(rgb_max, extractedVal - 1);
//					}
//					else
//					{
//						rgb = saturateRGB(rgb_min, 1-extractedVal);
//					}
//
//
//					//float3 rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;
//
//					rgb = rgb * diffuse;
//
//					// vector from eye to isosurface
//					float3 position_viewCoordinate = position - eyePos;
//
//					// calculates the z-value
//					float z_dist = abs(dot(viewDir, position_viewCoordinate));
//
//					// calculate non-linear depth between 0 to 1
//					float depth = (f) / (f - n);
//					depth += (-1.0f / z_dist) * (f * n) / (f - n);
//
//
//
//					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
//
//					// write back color and depth into the texture (surface)
//					// stride size of 4 * floats for each texel
//					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
//					break;
//				}
//				if (!dispersionOptions.marching)
//				{
//					t = t + samplingRate;
//				}
//				//else
//				//{
//				//	t = t + fmin(samplingRate, findExitPoint(make_float2(position.x, position.z), make_float2(rayDir.x, rayDir.z), cellSize_XZ));
//
//				//}
//
//			}
//
//		}
//
//	}
//}



//__global__ void CudaTerrainRenderer_Marching_extra_FTLE_Color
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t heightField,
//	cudaTextureObject_t extraField,
//	int rays,
//	float samplingRate,
//	float IsosurfaceTolerance,
//	DispersionOptions dispersionOptions
//)
//{
//	int index = CUDA_INDEX;
//
//	if (index < rays)
//	{
//
//		// determine pixel position based on the index of the thread
//		int2 pixel;
//		pixel.y = index / d_boundingBox.m_width;
//		pixel.x = index - pixel.y * d_boundingBox.m_width;
//
//		// copy values from constant memory to local memory (which one is faster?)
//		float3 viewDir = d_boundingBox.m_viewDir;
//		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
//		float2 t_range = findIntersections(pixelPos, d_boundingBox);
//
//
//		// if inside the bounding box
//		if (t_range.y != -1)
//		{
//
//			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
//
//			// near and far plane
//			float n = 0.1f;
//			float f = 1000.0f;
//
//			// Add the offset to the eye position
//			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
//			float t = t_range.x;
//
//
//			// While the ray is within the Bounding Box
//			while (t < t_range.y)
//			{
//
//				// Position of the isosurface
//				float3 position = pixelPos + (rayDir * t);
//
//				// Adds an offset to position while the center of the grid is at gridDiamter/2
//				position += d_boundingBox.gridDiameter / 2.0;
//
//				//Relative position calculates the position of the point on the cuda texture
//				float3 relativePos = world2Tex
//				(
//					make_float3(position.x, position.z, static_cast<float>(dispersionOptions.timestep)),
//					make_float3(d_boundingBox.gridDiameter.x, d_boundingBox.gridDiameter.z, d_boundingBox.gridSize.z),
//					d_boundingBox.gridSize
//				);
//
//				// fetch texel from the GPU memory
//				float4 ftle = ValueAtXYZ_float4(extraField, relativePos);
//				ftle.x *= dispersionOptions.scale;
//
//
//				// check if we have a hit 
//				if (position.y - ftle.x > 0 && position.y - ftle.x < dispersionOptions.hegiht_tolerance)
//				{
//					float3 gradient = normalize(GradientAtXYZ_Tex_X_Height(extraField, relativePos));
//					gradient = normalize(make_float3(gradient.x, -1.0f, gradient.y));
//
//					// shading (no ambient)
//					float diffuse = max(dot(gradient, viewDir), 0.0f);
//					float height = ValueAtXYZ_float4(extraField, relativePos).x;
//
//
//					float3 rgb_min = Array2Float3(dispersionOptions.minColor);
//					float3 rgb_max = Array2Float3(dispersionOptions.maxColor);
//					float3 rgb = rgb_min;
//
//
//					rgb = rgb * diffuse;
//
//					// vector from eye to isosurface
//					float3 position_viewCoordinate = position - eyePos;
//
//					// calculates the z-value
//					float z_dist = abs(dot(viewDir, position_viewCoordinate));
//
//					// calculate non-linear depth between 0 to 1
//					float depth = (f) / (f - n);
//					depth += (-1.0f / z_dist) * (f * n) / (f - n);
//
//
//
//					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
//
//					// write back color and depth into the texture (surface)
//					// stride size of 4 * floats for each texel
//					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);
//					break;
//				}
//				if (!dispersionOptions.marching)
//				{
//					t = t + samplingRate;
//				}
//
//			}
//
//		}
//
//	}
//}




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



