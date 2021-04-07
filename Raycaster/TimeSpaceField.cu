#include "TimeSpaceField.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/helper_math.h"


void TimeSpaceField::setResources
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
	TimeSpace3DOptions* _timeSpace3DOptions
)
{
	Raycasting::setResources(_camera, _width, _height, _solverOption, _raycastingOptions, _renderingOptions, _device, _pAdapter, _deviceContext);
	this->timeSpace3DOptions = _timeSpace3DOptions;
}

void TimeSpaceField::show(RenderImGuiOptions* renderImGuiOptions)
{
	if (renderImGuiOptions->showTimeSpaceField)
	{
		if (!timeSpace3DOptions->initialized)
		{
			this->initialize();
			timeSpace3DOptions->initialized = true;
		}
		
		// Override draw
		this->draw();

		if (renderImGuiOptions->updateTimeSpaceField)
		{
			this->updateScene();
			renderImGuiOptions->updateTimeSpaceField = false;

		}
	}
}

bool TimeSpaceField::generateVolumetricField(
	cudaTextureAddressMode addressMode_X ,
	cudaTextureAddressMode addressMode_Y ,
	cudaTextureAddressMode addressMode_Z
)
{
	generatedFieldSize =
	{
		solverOptions->gridSize[2],
		solverOptions->gridSize[1],
		timeSpace3DOptions->t_last - timeSpace3DOptions->t_first + 1
	};

	// initialize the 3D array
	if (!this->cudaArray3D.initialize(generatedFieldSize.x, generatedFieldSize.y, generatedFieldSize.z))
		return false;

	// bind it to the surface
	this->cudaSurface.setInputArray(cudaArray3D.getArrayRef());

	// initialize the surface
	if (!this->cudaSurface.initializeSurface())
		return false;


	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };

	blocks = BLOCK_THREAD(generatedFieldSize.x * generatedFieldSize.y);

	for (int i = 0; i < generatedFieldSize.z; i++)
	{
		volume_IO.readVolume(timeSpace3DOptions->t_first + i);
		volumeTexture.setField(volume_IO.getField_float());
		volumeTexture.initialize(Array2Int3(solverOptions->gridSize), true, cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);

		generateVorticityFieldSpaceTime << < blocks, thread >> >
			(
				cudaSurface.getSurfaceObject(),
				volumeTexture.getTexture(),
				*solverOptions,
				*timeSpace3DOptions,
				i,
				generatedFieldSize
			);
		volumeTexture.release();
	}

	traced = true;

	cudaSurface.destroySurface();
	cudaTexture3D_float.setArray(cudaArray3D.getArrayRef());
	cudaTexture3D_float.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);


	return true;
}


bool TimeSpaceField::regenerateVolumetricField()
{

	return true;
}




__global__ void  generateVorticityFieldSpaceTime
(
	cudaSurfaceObject_t s_vorticity,
	cudaTextureObject_t t_velocityField,
	SolverOptions solverOptions,
	TimeSpace3DOptions timeSpace3DOptions,
	int timestep,
	int3 gridSize
)
{
	// Extract dispersion options
	int kernelRun = gridSize.x * gridSize.y;

	int index = CUDA_INDEX;

	if (index < kernelRun)
	{

		// find the index of the particle
		int index_x = index / gridSize.y; // spanwise
		int index_y = index - (index_x * gridSize.y); //wall normal

		float3 h = Array2Float3(solverOptions.volumeDiameter) / Array2Int3(solverOptions.gridSize);

		//compute vorticity
		 fMat3X3 jac = Jacobian(t_velocityField, h, make_float3(timeSpace3DOptions.streamwisePos, index_y, index_x));
		 float3 vorticity_vec = make_float3
		 (
			 jac.r1.z - jac.r3.y,
			 jac.r3.x - jac.r1.z,
			 jac.r1.y - jac.r2.x
		 );
		 float velocity		= tex3D<float>(t_velocityField, timeSpace3DOptions.streamwisePos, index_y, index_x);
		 float vorticity	= magnitude(vorticity_vec);
		// copy it in the surface3D
		surf3Dwrite(vorticity, s_vorticity, sizeof(float) * index_x, index_y, timestep);
		//surf3Dwrite(vorticity, s_velocity, sizeof(float) * index_x, index_y, timestep);

	}
}


bool TimeSpaceField::cudaRaycaster()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));


	//We need normal raycasting here !!!!!!!
	//CudaIsoSurfacRenderer_float_PlaneColor << < blocks, thread >> >
	//	(
	//		raycastingSurface.getSurfaceObject(),
	//		this->cudaTexture3D_float.getTexture(),
	//		int(this->rays),
	//		generatedFieldSize,
	//		*timeSpace3DOptions
	//		);

	return true;
}



