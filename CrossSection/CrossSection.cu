#include "CrossSection.h"
#include "../ErrorLogger/ErrorLogger.h"
#include "../Raycaster/Raycasting.h"
#include "../Raycaster/IsosurfaceHelperFunctions.h"
#include "../Cuda/helper_math.h"
#include <vector>

// Explicit specialization
template <> void CrossSection::traceCrossSectionField< CrossSectionOptionsMode::SpanMode::TIME>();
template <> void CrossSection::traceCrossSectionField< CrossSectionOptionsMode::SpanMode::WALL_NORMAL>();

bool CrossSection::initialize
(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z
)  
{
	if (!this->initializeRaycastingTexture())		// initilize texture (the texture we need to write to)
		return false;

	if (!this->initializeBoundingBox())				// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;
	
	// set the number of rays = number of pixels
	this->rays = size_t(*this->width) * size_t(*this->height);

	volume_IO.Initialize(this->solverOptions);

	if (this->crossSectionOptions->mode == CrossSectionOptionsMode::SpanMode::WALL_NORMAL)
	{
		traceCrossSectionField<CrossSectionOptionsMode::SpanMode::WALL_NORMAL>();

	}
	else
	{
		traceCrossSectionField<CrossSectionOptionsMode::SpanMode::TIME>();
	}

	return true;
}



void CrossSection::retraceCrossSectionField()
{
	this->t_volumeTexture.release();
	this->volume_IO.readVolume(solverOptions->currentIdx);		// Read a velocity volume
	t_volumeTexture.setField(volume_IO.getField_float());	// Pass a pointer to the Cuda volume texture
	t_volumeTexture.initialize(Array2Int3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);					// Initilize the Cuda texture

	volume_IO.release();										// Release velocity volume from host memory
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


	if (this->crossSectionOptions->mode == CrossSectionOptionsMode::SpanMode::WALL_NORMAL)
	{
		CudaCrossSectionRenderer<CrossSectionOptionsMode::SpanMode::WALL_NORMAL> << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->t_volumeTexture.getTexture(),
				this->t_volumeGradient.getTexture(),
				int(this->rays),
				crossSectionOptions->samplingRate,
				this->raycastingOptions->tolerance_0,
				*solverOptions,
				*crossSectionOptions
				);

	}
	else if(this->crossSectionOptions->mode == CrossSectionOptionsMode::SpanMode::TIME)
	{
		if (crossSectionOptions->filterMinMax)
		{
			CudaCrossSectionRenderer<CrossSectionOptionsMode::SpanMode::TIME> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->t_volumeTexture.getTexture(),
					this->t_volumeGradient.getTexture(),
					int(this->rays),
					crossSectionOptions->samplingRate,
					this->raycastingOptions->tolerance_0,
					*solverOptions,
					*crossSectionOptions
					);
		}
		else
		{
			CudaCrossSectionRenderer<CrossSectionOptionsMode::SpanMode::TIME> << < blocks, thread >> >
				(
					this->raycastingSurface.getSurfaceObject(),
					this->t_volumeTexture.getTexture(),
					this->t_volumeGradient.getTexture(),
					int(this->rays),
					crossSectionOptions->samplingRate,
					this->raycastingOptions->tolerance_0,
					*solverOptions,
					*crossSectionOptions
				);
		}

	}

	else if (this->crossSectionOptions->mode == CrossSectionOptionsMode::SpanMode::VOL_3D)
	{
		CudaIsoSurfacRendererSpaceTime<FetchTextureSurface::Channel_X> << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->t_volumeTexture.getTexture(),
				int(this->rays),
				this->raycastingOptions->isoValue_0,
				this->raycastingOptions->samplingRate_0,
				this->raycastingOptions->tolerance_0
				);
	}


}


// Specialization for wall-normal span
template <> void CrossSection::traceCrossSectionField< CrossSectionOptionsMode::SpanMode::WALL_NORMAL>()
{

	
	this->volume_IO.readVolume(solverOptions->currentIdx);		// Read a velocity volume
	t_volumeTexture.setField(volume_IO.getField_float());	// Pass a pointer to the Cuda volume texture
	
	t_volumeTexture.initialize(ARRAYTOINT3(solverOptions->gridSize), false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);								// Initilize the Cuda texture

	volume_IO.release();


}

// Specialization for time span
template <> void CrossSection::traceCrossSectionField< CrossSectionOptionsMode::SpanMode::TIME>()
{

	// First Read XY plane
	int t0 = this->solverOptions->firstIdx;
	int t1 = this->solverOptions->lastIdx;
	int dt = t1 - t0 + 1;

	m_dimension = { solverOptions->gridSize[2], solverOptions->gridSize[0], dt };

	float* h_velocity = new float[dt * solverOptions->gridSize[0] * solverOptions->gridSize[2] * 4];
	size_t pass = 0;
	for (int t = t0; t <= t1; t++)
	{
		for (int x = 0; x < solverOptions->gridSize[0]; x++)
		{

			//volume_IO_X_Major.readVolumePlane(t,x,crossSectionOptions->wallNormalPos);

			for (int i = 0; i < solverOptions->gridSize[2] * 4; i++)
			{
				h_velocity[i + pass] = volume_IO.getField_float()[i];
			}
			pass += solverOptions->gridSize[2] * 4;
			volume_IO.release();
		}

	}


	t_volumeTexture.setField(h_velocity);


	


	if (crossSectionOptions->filterMinMax)
	{
		t_volumeTexture.initialize(m_dimension,false,cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder,cudaFilterModePoint);
		
		initializedFilterSurface();
		filterExtermum();
		cudaArray_t pCudaArray = a_field.getArray();
		t_volumeGradient.setArray(pCudaArray);
		t_volumeGradient.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);

		cudaDestroyTextureObject(t_volumeTexture.getTexture());
		t_volumeTexture.initialize(m_dimension,false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);

		delete[] h_velocity;
	}
	else
	{
		t_volumeTexture.initialize(m_dimension, false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
		delete[] h_velocity;
	}
	
}


void CrossSection::filterExtermum()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	size_t points = (size_t)m_dimension.x * (size_t)m_dimension.y;
	blocks = static_cast<unsigned int>((points % (thread.x * thread.y) == 0 ? points / (thread.x * thread.y) : points / (thread.x * thread.y) + 1));

	int2 dimension = { m_dimension.x, m_dimension.y };
	float threshold = 0.1f;
	for (int t = 0; t < m_dimension.z; t++)
	{
		CudaFilterExtremumX <<< blocks, thread >> >
			(
				s_filteringSurface.getSurfaceObject(),
				t_volumeTexture.getTexture(),
				dimension,
				threshold,
				t
			);
	}
	s_filteringSurface.destroySurface();

}


void CrossSection::initializedFilterSurface()
{
	this->a_field.initialize(m_dimension.x, m_dimension.y, m_dimension.z);

	cudaArray_t pCudaArray = a_field.getArray();
	this->s_filteringSurface.setInputArray(pCudaArray);
	this->s_filteringSurface.initializeSurface();
}