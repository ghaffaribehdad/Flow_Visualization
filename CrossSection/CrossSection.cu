#include "CrossSection.h"
#include "../ErrorLogger/ErrorLogger.h"
#include "../Raycaster/Raycasting.h"
#include "../Raycaster/IsosurfaceHelperFunctions.h"

bool CrossSection::initialize
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
	this->rays = size_t(*this->width) * size_t(*this->height);

	primary_IO.Initialize(this->solverOptions);

	traceCrossSectionField();

	return true;
}

void CrossSection::traceCrossSectionField()
{

	this->primary_IO.readVolume(solverOptions->currentIdx);		// Read a velocity volume
	t_volumeTexture.setField(primary_IO.flushBuffer_float());	// Pass a pointer to the Cuda volume texture
	t_volumeTexture.setSolverOptions(this->solverOptions);
	t_volumeTexture.initialize();								// Initilize the Cuda texture

	primary_IO.release();										// Release velocity volume from host memory
}




void CrossSection::setResources
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
	CrossSectionOptions* _crossSectionOptions
)
{
	Raycasting::setResources(_camera, _width, _height, _solverOption, _raycastingOptions, _renderingOptions, _device, _pAdapter, _deviceContext);
	this->crossSectionOptions = _crossSectionOptions;
}



__host__ void CrossSection::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());

	// Create a 2D texture to read hight array


	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));


	CudaCrossSectionRenderer<IsosurfaceHelper::Velocity_XYZT> << < blocks, thread >> >
		(
			this->raycastingSurface.getSurfaceObject(),
			this->t_volumeTexture.getTexture(),
			int(this->rays),
			crossSectionOptions->samplingRate,
			this->raycastingOptions->tolerance_0,
			*crossSectionOptions
			);

}