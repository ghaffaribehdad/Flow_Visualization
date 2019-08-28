#include "Raycasting.h"
#include "IsosurfaceHelperFunctions.h"
#include "Raycasting_Helper.h"


__constant__ BoundingBox d_boundingBox;


// Explicit instantiation
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Magnitude>	(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_X>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Y>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Z>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);



__host__ bool Raycasting::updateScene()
{
	if (!this->initializeRaycastingInteroperability())	// Create interoperability while we need to release it at the end of rendering
		return false;

	if (!this->initializeCudaSurface())					// reinitilize cudaSurface	
		return false;

	if (!this->initializeBoundingBox())	//updates constant memory
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

__host__ bool Raycasting::initialize()
{
	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;

	//if (!this->initializeRaycastingInteroperability())		// initilize interoperability (the pointer to access texture via CUDA)
	//	return false;

	//if (!this->initializeCudaSurface())		// initilize CUDA surface ( an interface to write into the cuda array obtained by interoperability)
	//	return false;

	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);	// Set number of rays based on the number of pixels


	// Read and set field
	if(!this->raycastingOptions->fileLoaded)		// Load data set into the texture memory
	{
		this->volume_IO.Initialize(this->solverOptions);
		this->initializeIO();
		this->initializeVolumeTexuture();

		this->raycastingOptions->fileLoaded = true;
	}
	if (this->raycastingOptions->fileChanged)
	{
		this->initializeIO();
		this->volumeTexture.release();
		this->initializeVolumeTexuture();

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
	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view

	this->initializeBoundingBox();

	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));


	CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Magnitude> <<< blocks, thread >> >
	(
		this->raycastingSurface.getSurfaceObject(),
		this->volumeTexture.getTexture(),
		int(this->rays),
		this->raycastingOptions->isoValue_0,
		this->raycastingOptions->samplingRate_0,
		this->raycastingOptions->tolerance_0
	);


}



__host__ bool Raycasting::initializeBoundingBox()
{

	BoundingBox * h_boundingBox = new BoundingBox;

	h_boundingBox->eyePos = XMFloat3ToFloat3(camera->GetPositionFloat3());
	h_boundingBox->viewDir = XMFloat3ToFloat3(camera->GetViewVector());
	h_boundingBox->upVec = XMFloat3ToFloat3(camera->GetUpVector());


	h_boundingBox->width = *width;
	h_boundingBox->height= *height;
	h_boundingBox->gridDiameter = ArrayFloat3ToFloat3(solverOptions->gridDiameter);
	h_boundingBox->updateBoxFaces();
	h_boundingBox->updateAspectRatio();
	h_boundingBox->constructEyeCoordinates();
	h_boundingBox->FOV = (30.0f) * 3.1415f / 180.0f;
	h_boundingBox->distImagePlane = 1;

	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));
	
	delete h_boundingBox;
	
	return true;
}


__host__ bool Raycasting::initializeVolumeTexuture()
{
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.setField(this->field);
	this->volumeTexture.initialize();

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
__global__ void CudaIsoSurfacRenderer(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance)
{

	Observable observable;

	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < rays)
	{


		int2 pixel;
		pixel.y = index / d_boundingBox.width;
		pixel.x = index - pixel.y * d_boundingBox.width;
		float3 viewDir = d_boundingBox.viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		float3 rgb = { 0.5f, 0.5f, 0.5f };

		float3 positionOffset = d_boundingBox.gridDiameter / 2.0f;
		// if hits
		if (NearFar.y != -1)
		{
			float3 rayDir = normalize(pixelPos - d_boundingBox.eyePos);

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{

				float3 relativePos = pixelPos + (rayDir * t);
				relativePos = (relativePos / d_boundingBox.gridDiameter) + make_float3(.5f, .5f, .5f);
				float4 velocity4D = tex3D<float4>(field1, relativePos.x, relativePos.y, relativePos.z);

				if (fabsf(observable.ValueAtXYZ(field1, relativePos) - isoValue) < IsosurfaceTolerance)
				{
					float3 gradient = observable.GradientAtXYZ(field1, relativePos, 0.01f);
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
					rgb = rgb * diffuse;
					float4 rgba = { rgb.x,rgb.y,rgb.z,1 };
										
					surf2Dwrite(rgbaFloatToUChar(rgba), raycastingSurface, 4 * pixel.x, pixel.y);
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
	//textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

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
	//interoperability_desc.size = 4 * sizeof(float) * static_cast<size_t>(this->width) * static_cast<size_t>(this->height);
	interoperability_desc.size = sizeof(float) * static_cast<size_t>(*this->width) * static_cast<size_t>(*this->height);
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

	this->raycastingSurface.setDimensions(*this->width, *this->height);

	// Create cuda surface 
	if (!this->raycastingSurface.initializeSurface())
		return false;

	// To release we need to destory surface and free the cuda array kept in the interpoly

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



