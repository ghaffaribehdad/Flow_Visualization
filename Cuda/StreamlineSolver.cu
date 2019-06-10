#include "StreamlineSolver.cuh"


// Explicit instantions
template class StreamlineSolver<float>;
template class StreamlineSolver<double>;



template<typename T>
__host__ T* StreamlineSolver<T>::InitializeVelocityField()
{
	std::vector<char>* p_vec_buffer = this->volume_IO.readVolume(0);

	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));

	this->h_VelocityField = reinterpret_cast<T*> (p_vec_buffer_temp);

	size_t velocityField_byte =
		solverOptions.gridSize[0] *
		solverOptions.gridSize[1] *
		solverOptions.gridSize[2] * 3 * sizeof(T);

	// Allocate memory on Device
	gpuErrchk(cudaMalloc((void**)& d_VelocityField, velocityField_byte));

	// Upload Velocity Filled to GPU 
	gpuErrchk(cudaMemcpy(d_VelocityField, h_VelocityField, velocityField_byte, cudaMemcpyHostToDevice));

	return h_VelocityField;
}

template <typename T>
void StreamlineSolver<T>::InitializeParticles()
{
	// Create an array of particles
	this->h_Particles = new Particle<T>[solverOptions.particle_count];

	float3 gridDiameter =
	{
		solverOptions.gridDiameter[0],
		solverOptions.gridDiameter[1],
		solverOptions.gridDiameter[2]
	};
	// Seed Particles Randomly according to the grid diameters
	for (int i = 0; i < solverOptions.particle_count; i++)
	{
		h_Particles[i].seedParticle(gridDiameter);
	}
	
	size_t Particles_byte = sizeof(*h_Particles) * solverOptions.particle_count;

	// Upload Velocity Filled to GPU 

	gpuErrchk(cudaMalloc((void**)& d_Particles, Particles_byte));

	gpuErrchk(cudaMemcpy(d_Particles, h_Particles, Particles_byte, cudaMemcpyHostToDevice));

	delete h_Particles;
}


template <typename T>
__host__ bool StreamlineSolver<T>::solve()
{
	float * h_VelocityField = InitializeVelocityField();
	this->InitializeTexture(h_VelocityField);
	this->InitializeParticles();

	TracingParticles<T> << < 1, solverOptions.particle_count >> > (d_Particles, d_VelocityField, solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));
	//TracingParticles<T> << < 1, solverOptions.particle_count >> > (d_Particles, t_VelocityField, solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	cudaFree(d_Particles);
	cudaFree(d_VelocityField);
	return true;
}


template <typename T>
__host__ bool StreamlineSolver<T>::InitializeTexture(T* h_VelocityField)
{

	// Cuda 3D array of velocities
	cudaArray_t cuArray_velocity;


	// define the size of the velocity field
	cudaExtent extent = 
	{ 
		solverOptions.gridSize[0],
		solverOptions.gridSize[1],
		solverOptions.gridSize[2] 
	};


	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	gpuErrchk(cudaMalloc3DArray(&cuArray_velocity, &channelFormatDesc, extent, 0));



	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcPtr = make_cudaPitchedPtr((void*)h_VelocityField, extent.width * sizeof(float4), extent.height, extent.depth);
	cpyParams.dstArray = cuArray_velocity;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cpyParams.extent = extent;


	// Copy velocities to 3D Array
	gpuErrchk(cudaMemcpy3D(&cpyParams));



	//// Create a texture
	cudaTextureObject_t t_VelocityField = 0;

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
	texDesc.filterMode = cudaFilterModeLi;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;



	gpuErrchk(cudaCreateTextureObject(&t_VelocityField, &resDesc, &texDesc, NULL));

	//cudaDestroyTextureObject(t_VelocityField);
	return true;

}