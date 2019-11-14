#include "Raycasting.h"
#include "IsosurfaceHelperFunctions.h"
#include "Raycasting_Helper.h"
#include "cuda_runtime.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Options/DispresionOptions.h"

__constant__   BoundingBox d_boundingBox; // constant memory variable
__constant__   float3 d_raycastingColor;

// Explicit instantiation
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Magnitude>	(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_X>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Y>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Z>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::ShearStress>		(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaTerrainRenderer< struct IsosurfaceHelper::Position >			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float samplingRate, float IsosurfaceTolerance, DispersionOptions dispersionOptions);
template __global__ void CudaTerrainRenderer_extra< struct IsosurfaceHelper::Position >(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t extraField, int rays, float samplingRate, float IsosurfaceTolerance, DispersionOptions dispersionOptions);
template __global__ void CudaTerrainRenderer_extra_fluctuation< struct IsosurfaceHelper::Position >(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t heightField, cudaTextureObject_t extraField, int rays, float samplingRate, float IsosurfaceTolerance, FluctuationheightfieldOptions fluctuationOptions);

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

__host__ bool Raycasting::resize()
{
	this->raycastingTexture->Release();
	this->initializeRaycastingTexture();

	this->raycastingSurface.destroySurface();
	this->interoperatibility.release();

	this->initializeRaycastingInteroperability();
	this->initializeCudaSurface();
	this->initializeBoundingBox();
	this->rendering();
	this->interoperatibility.release();

	return true;
}

__host__ bool Raycasting::initialize
(
	cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
	cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
	cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
)
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels


	// Read and set field
	if(!this->raycastingOptions->fileLoaded)		// Load data set into the texture memory
	{
		this->volume_IO.Initialize(this->solverOptions);
		this->initializeIO();
		this->initializeVolumeTexuture(addressMode_X, addressMode_Y, addressMode_Z);

		this->raycastingOptions->fileLoaded = true;
	}
	if (this->raycastingOptions->fileChanged)
	{
		this->initializeIO();
		this->volumeTexture.release();
		this->initializeVolumeTexuture(addressMode_X, addressMode_Y, addressMode_Z);

		this->raycastingOptions->fileChanged = false;
	}

	return true;

}

__host__ bool Raycasting::release()
{
	this->interoperatibility.release();
	this->volumeTexture.release();
	this->raycastingSurface.destroySurface();

	return true;
}

__host__ void Raycasting::rendering()
{

	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());



	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { 32,32,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));




	// Optimize blocks and grid sizes
	//int* minGridSize	= nullptr;
	//int* blockSize		= nullptr;	
	//cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize,(CUfunction)CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Magnitude>, 0, 0,0);



	// TODO:
	// Alternatively use ENUM templates! 
	switch (this->raycastingOptions->isoMeasure_0)
	{
		case IsoMeasure::VelocityMagnitude:
		{
			CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Magnitude> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture.getTexture(),
					int(this->rays),
					this->raycastingOptions->isoValue_0,
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0
					);
			break;
		}

		case IsoMeasure::Velocity_x:
		{
			CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_X> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture.getTexture(),
					int(this->rays),
					this->raycastingOptions->isoValue_0,
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0
				);
			break;

		}

		case IsoMeasure::Velocity_y:
		{
			CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Y> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture.getTexture(),
					int(this->rays),
					this->raycastingOptions->isoValue_0,
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0
				);
			break;
		}

		case IsoMeasure::Velocity_Z:
		{
			CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Z> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture.getTexture(),
					int(this->rays),
					this->raycastingOptions->isoValue_0,
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0
				);
			break;
		}

		case IsoMeasure::ShearStress:
		{
			CudaIsoSurfacRenderer<IsosurfaceHelper::ShearStress> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->volumeTexture.getTexture(),
					int(this->rays),
					this->raycastingOptions->isoValue_0,
					this->raycastingOptions->samplingRate_0,
					this->raycastingOptions->tolerance_0
				);
			break;
		}

	}



}



__host__ bool Raycasting::initializeBoundingBox()
{

	BoundingBox * h_boundingBox = new BoundingBox;
	

	h_boundingBox->gridSize = ArrayInt3ToInt3(solverOptions->gridSize);
	h_boundingBox->updateBoxFaces(ArrayFloat3ToFloat3(solverOptions->gridDiameter));
	h_boundingBox->updateAspectRatio(*width,*height);

	h_boundingBox->constructEyeCoordinates
	(
		XMFloat3ToFloat3(camera->GetPositionFloat3()),
		XMFloat3ToFloat3(camera->GetViewVector()), 
		XMFloat3ToFloat3(camera->GetUpVector())
	);

	h_boundingBox->FOV = (this->FOV_deg / 360.0f)* XM_2PI;
	h_boundingBox->distImagePlane = this->distImagePlane;
	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));
	
	gpuErrchk(cudaMemcpyToSymbol(d_raycastingColor, this->raycastingOptions->color_0, sizeof(float3)));


	delete h_boundingBox;
	
	return true;
}


__host__ bool Raycasting::initializeVolumeTexuture
(
	cudaTextureAddressMode addressMode_X ,
	cudaTextureAddressMode addressMode_Y ,
	cudaTextureAddressMode addressMode_Z
	)
{
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.setField(this->field);
	this->volumeTexture.initialize
	(
		addressMode_X,
		addressMode_Y,
		addressMode_Z
	);

	return true;
}


__host__  bool Raycasting::initializeVolumeTexuture
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z,
	VolumeTexture3D & _volumeTexture
)
{
	_volumeTexture.setSolverOptions(this->solverOptions);
	_volumeTexture.setField(this->field);
	_volumeTexture.initialize
	(
		addressMode_X,
		addressMode_Y,
		addressMode_Z
	);

	return true;
}

__host__ bool Raycasting::initializeIO()
{
	
	this->volume_IO.readVolume(this->solverOptions->currentIdx);
	std::vector<char>* p_vec_buffer = volume_IO.flushBuffer();
	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));
	this->field = reinterpret_cast<float*>(p_vec_buffer_temp);
	
	return true;
}


 




template <typename Observable>
__global__ void CudaIsoSurfacRenderer
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
				if (observable.ValueAtXYZ(field1, relativePos) - isoValue > 0)
				{

					position = binarySearch<Observable>(observable, field1, position, d_boundingBox.m_dimensions, rayDir * t, isoValue, IsosurfaceTolerance, 50);
					relativePos = (position / d_boundingBox.m_dimensions);

					// calculates gradient
					float3 gradient = observable.GradientAtGrid(field1, relativePos, d_boundingBox.gridSize);

					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
					float3 rgb = d_raycastingColor * diffuse;


					// vector from eye to isosurface
					float3 position_viewCoordinate = position - eyePos;

					// calculates the z-value
					float z_dist = abs(dot(viewDir, position_viewCoordinate));

					// calculate non-linear depth between 0 to 1
					float depth = (f) / (f - n);
					depth += (-1.0f / z_dist) * (f * n) / (f - n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth};
					
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
	DispersionOptions dispersionOptions
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
				float3 relativePos = 
				{ 
					position.x / d_boundingBox.m_dimensions.x,
					position.z / d_boundingBox.m_dimensions.z,
					static_cast<float> (dispersionOptions.timeStep) / static_cast<float> (dispersionOptions.tracingTime)
				};
				
				// fetch texels from the GPU memory
				float4 hightFieldVal = observable.ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 &&  position.y - hightFieldVal.x < 0.01 )
				{

					float3 samplingStep = rayDir * samplingRate;
					//binary search
					//position = binarySearch_heightField
					//(
					//	position,
					//	field1,
					//	samplingStep,
					//	d_boundingBox.gridDiameter,
					//	dispersionOptions.binarySearchTolerance,
					//	dispersionOptions.binarySearchMaxIteration
					//);

					relativePos = make_float3
					(
						position.x / d_boundingBox.m_dimensions.x,
						position.z / d_boundingBox.m_dimensions.z,
						static_cast<float> (dispersionOptions.timeStep) / static_cast<float> (dispersionOptions.tracingTime)
					);

					hightFieldVal = observable.ValueAtXYZ_float4(heightField, relativePos);

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




template <typename Observable>
__global__ void CudaTerrainRenderer_extra
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
				float3 relativePos =
				{
					position.x / d_boundingBox.m_dimensions.x,
					position.z / d_boundingBox.m_dimensions.z,
					static_cast<float> (dispersionOptions.timeStep) / static_cast<float> (dispersionOptions.tracingTime)
				};

				// fetch texels from the GPU memory
				float4 hightFieldVal = observable.ValueAtXYZ_float4(heightField, relativePos);

				// check if we have a hit 
				if (position.y - hightFieldVal.x > 0 && position.y - hightFieldVal.x < dispersionOptions.hegiht_tolerance)
				{

		
					hightFieldVal = observable.ValueAtXYZ_float4(heightField, relativePos);
					
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
						case dispersionColorCode::NONE:
						{
							break;
						}

						case dispersionColorCode::V_X_FLUCTUATION:
						{

							float4 value = observable.ValueAtXYZ_float4(extraField, relativePos);
							extractedVal = value.y;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionColorCode::V_Y:
						{

							extractedVal = observable.ValueAtXYZ_float4(extraField, relativePos).z;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionColorCode::V_Z:
						{

							extractedVal = observable.ValueAtXYZ_float4(extraField, relativePos).w;
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));
							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionColorCode::DISTANCE:
						{
							float4 Vec_0 = observable.ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y,0.0));
							float4 Vec_1 = observable.ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y,0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = observable.ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = observable.ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { Vec_0.w,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionColorCode::DISTANCE_ZY:
						{
							float4 Vec_0 = observable.ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = observable.ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { 0.0f,Vec_0.x,Vec_1.x };

							Vec_0 = observable.ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = observable.ValueAtXYZ_float4(extraField, relativePos);

							float3 position_1 = { 0.0f,Vec_0.x,Vec_1.x };

							extractedVal = sqrtf(dot(position_1 - position_0, position_1 - position_0));
							extractedVal = saturate((extractedVal - dispersionOptions.min_val) / (dispersionOptions.max_val - dispersionOptions.min_val));

							rgb = (1.0f - extractedVal) * rgb_min + extractedVal * rgb_max;

							break;
						}

						case dispersionColorCode::DEV_Z:
						{
							float4 Vec_1 = observable.ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float position_0 = Vec_1.x;

							Vec_1 = observable.ValueAtXYZ_float4(extraField, relativePos);

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
						case dispersionColorCode::QUADRANT_DEV:
						{
							float4 Vec_0 = observable.ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = observable.ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = observable.ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = observable.ValueAtXYZ_float4(extraField, relativePos);

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



						case dispersionColorCode::DEV_ZY:
						{
							float4 Vec_0 = observable.ValueAtXYZ_float4(heightField, make_float3(relativePos.x, relativePos.y, 0.0));
							float4 Vec_1 = observable.ValueAtXYZ_float4(extraField, make_float3(relativePos.x, relativePos.y, 0.0));

							float3 position_0 = { Vec_0.w,Vec_0.x,Vec_1.x };

							Vec_0 = observable.ValueAtXYZ_float4(heightField, relativePos);
							Vec_1 = observable.ValueAtXYZ_float4(extraField, relativePos);

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

	this->device = _device;
	this->pAdapter = _pAdapter;
	this->deviceContext = _deviceContext;
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
		rasterizerDesc.MultisampleEnable = false;
		rasterizerDesc.AntialiasedLineEnable = false;
		//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;

		HRESULT hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
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
}


void Raycasting::draw()
{
	this->initializeRasterizer();
	this->initializeSamplerstate();
	this->createRaycastingShaderResourceView();

	this->initializeShaders();
	this->initializeScene();


	this->setShaders();
	this->deviceContext->Draw(6, 0);
}






template <typename Observable>
__global__ void CudaTerrainRenderer_extra_fluctuation
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	FluctuationheightfieldOptions fluctuationOptions
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



		//  if inside the bounding box
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
					static_cast<float>(fluctuationOptions.wallNoramlPos)/ static_cast<float>(fluctuationOptions.wallNormalgridSize),
					position.x / d_boundingBox.m_dimensions.x
				};


				// fetch texels from the GPU memory

				float4 hightFieldVal = tex3D<float4>(heightField, relativePos.x, relativePos.y, relativePos.z);
				float height = 0;

				if (fluctuationOptions.usingAbsolute)
				{
					height = abs(hightFieldVal.y * fluctuationOptions.height_scale + fluctuationOptions.offset);

				}
				else
				{
					height = (hightFieldVal.y * fluctuationOptions.height_scale) + fluctuationOptions.offset;
				}

				height = fluctuationOptions.heightLimit * saturate(height / fluctuationOptions.heightLimit);

				if (position.y - height > 0 && position.y - height < fluctuationOptions.hegiht_tolerance)
				{
					float value = hightFieldVal.x;
				

					

					float3 samplingStep = rayDir * samplingRate;

					float3 gradient = { hightFieldVal.w,-1, hightFieldVal.z };


					// shading (no ambient)
					float diffuse = max(dot(gradient, viewDir), 0.0f);



					float3 rgb_min =
					{
						fluctuationOptions.minColor[0],
						fluctuationOptions.minColor[1],
						fluctuationOptions.minColor[2],
					};

					float3 rgb_max =
					{
						fluctuationOptions.maxColor[0],
						fluctuationOptions.maxColor[1],
						fluctuationOptions.maxColor[2],
					};

					float3 rgb = { 0,0,0 };
					float y_saturated = 0.0f;

					if (height == fluctuationOptions.heightLimit)
					{
						if (value < 0)
						{
							float3 rgb_min_complement = make_float3(1, 1, 1) - rgb_min;
							y_saturated = saturate(abs(value / fluctuationOptions.min_val));
							rgb = rgb_min_complement * (1 - y_saturated) + rgb_min;
						}
						else
						{
							float3 rgb_max_complement = make_float3(1, 1, 1) - rgb_max;
							y_saturated = saturate(value / fluctuationOptions.max_val);
							rgb = rgb_max_complement * (1 - y_saturated) + rgb_max;
						}
					}
				


					//rgb = rgb * diffuse;

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