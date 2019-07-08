#include "CudaSolver.h"

template class CUDASolver<float>;
template class CUDASolver<double>;

template <typename T>
CUDASolver<T>::CUDASolver()
{
	std::printf("A solver is created!\n");
}

// Initilize the solver
template <typename T>
bool CUDASolver<T>::Initialize(SolverOptions _solverOptions)
{
	this->solverOptions = _solverOptions;
	this->InitializeCUDA();
	
	return true;
}


bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox)
{
	return true;
}


template <typename T>
bool CUDASolver<T>::FinalizeCUDA()
{
	gpuErrchk(cudaGraphicsUnmapResources(1,	&this->cudaGraphics	));

	gpuErrchk(cudaGraphicsUnregisterResource(this->cudaGraphics));

	return true;
}

template <typename T>
bool CUDASolver<T>::InitializeCUDA()
{
	// Get number of CUDA-Enable devices
	int device;
	gpuErrchk(cudaD3D11GetDevice(&device,solverOptions.p_Adapter));

	// Get properties of the Best(usually at slot 0) card
	gpuErrchk(cudaGetDeviceProperties(&this->cuda_device_prop, 0));

	// Register Vertex Buffer to map it
	gpuErrchk(cudaGraphicsD3D11RegisterResource(
		&this->cudaGraphics,
		this->solverOptions.p_vertexBuffer,
		cudaGraphicsRegisterFlagsNone));

	// Map Vertex Buffer
	gpuErrchk(cudaGraphicsMapResources(
		1,
		&this->cudaGraphics
		));

	// Get Mapped pointer
	size_t size = static_cast<size_t>(solverOptions.lines_count)* static_cast<size_t>(solverOptions.lineLength)*sizeof(Vertex);

	gpuErrchk(cudaGraphicsResourceGetMappedPointer(
		&p_VertexBuffer,
		&size,
		this->cudaGraphics
	));

	return true;
}

template<typename T>
__host__ T* CUDASolver<T>::InitializeVelocityField(int ID)
{
	this->volume_IO.readVolume(ID);
	std::vector<char>* p_vec_buffer = volume_IO.flushBuffer();
	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));


	return reinterpret_cast<T*>(p_vec_buffer_temp);
}


template <typename T1>
__host__ bool CUDASolver<T1>::InitializeTexture(T1* h_source, cudaTextureObject_t& texture)
{

	// Cuda 3D array of velocities
	cudaArray_t cuArray_velocity;


	// define the size of the velocity field
	cudaExtent extent =
	{
		static_cast<size_t>(solverOptions.gridSize[0]),
		static_cast<size_t>(solverOptions.gridSize[1]),
		static_cast<size_t>(solverOptions.gridSize[2])
	};


	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMalloc3DArray(&cuArray_velocity, &channelFormatDesc, extent, 0));



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)h_source, extent.width * sizeof(float4), extent.height, extent.depth);
	cpyParams.dstArray = cuArray_velocity;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cpyParams.extent = extent;


	// Copy velocities to 3D Array
	gpuErrchk(cudaMemcpy3D(&cpyParams));
	// might need sync before release the host memory

	// Release the Volume while it is copied on GPU
	this->volume_IO.release();


	// Set Texture Description
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;
	cudaResourceViewDesc resViewDesc;

	memset(&resDesc, 0, sizeof(resDesc));
	memset(&texDesc, 0, sizeof(texDesc));
	memset(&resViewDesc, 0, sizeof(resViewDesc));



	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray_velocity;

	// Texture Description
	texDesc.normalizedCoords = true;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;



	// Create the texture and bind it to the array
	gpuErrchk(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));

	return true;

}


template <typename T1>
void CUDASolver<T1>::InitializeParticles()
{
	// Create an array of particles
	this->h_Particles = new Particle<T1>[solverOptions.lines_count];

	float3 gridDiameter =
	{
		solverOptions.gridDiameter[0],
		solverOptions.gridDiameter[1],
		solverOptions.gridDiameter[2]
	};
	// Seed Particles Randomly according to the grid diameters
	for (int i = 0; i < solverOptions.lines_count; i++)
	{
		this->h_Particles[i].seedParticle(gridDiameter);
	}

	size_t Particles_byte = sizeof(Particle<T1>) * solverOptions.lines_count;

	// Upload Velocity Filled to GPU 

	gpuErrchk(cudaMalloc((void**) &this->d_Particles, Particles_byte));

	gpuErrchk(cudaMemcpy(this->d_Particles, this->h_Particles, Particles_byte, cudaMemcpyHostToDevice));

	delete this->h_Particles;
}

