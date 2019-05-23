#include "StreamlineSolver.cuh"

// Explicit instantions
template class StreamlineSolver<float>;
template class StreamlineSolver<double>;


template<typename T>
__host__ void StreamlineSolver<T>::InitializeVelocityField()
{
	// 1. Read and store Vector Field *OK
	h_velocityField = new VelocityField<T>;
	

	std::string fullPath = "";
	fullPath += std::string(this->solverOptions.filePath);
	fullPath += std::string(this->solverOptions.fileName);
	fullPath += std::to_string(this->solverOptions.timestep);
	fullPath += ".bin";

	std::vector<char>* p_vec_buffer = new std::vector<char>;


	// DEBGU:: path is fixed inside ReadField function
	this->ReadField(p_vec_buffer, fullPath);

	char* p_vec_buffer_temp = &(p_vec_buffer ->at(0));

	this->h_velocityField->setVelocityField(reinterpret_cast<T*> (p_vec_buffer_temp));

	// 2. Set the parameter of Vector Field *OK
	int3 gridSize;
	gridSize.x = this->solverOptions.gridSize[0];
	gridSize.y = this->solverOptions.gridSize[1];
	gridSize.z = this->solverOptions.gridSize[2];
	h_velocityField->setGridSize(gridSize);

	float3 gridDiameter;
	gridDiameter.x = this->solverOptions.gridDiameter[0];
	gridDiameter.y = this->solverOptions.gridDiameter[1];
	gridDiameter.z = this->solverOptions.gridDiameter[2];
	h_velocityField->setGridDiameter(gridDiameter);

	// 3. Upload Velocity Filled to GPU *OK
	UploadToGPU<VelocityField<T>>(this->d_velocityField,h_velocityField, sizeof(h_velocityField));

	delete p_vec_buffer;

}

template <typename T>
void StreamlineSolver<T>::InitializeParticles()
{
	// Create an array of particles
	this->h_particles = new Particle<T>[solverOptions.particle_count];

	float3 gridDiameter;
	gridDiameter.x = this->solverOptions.gridDiameter[0];
	gridDiameter.y = this->solverOptions.gridDiameter[1];
	gridDiameter.z = this->solverOptions.gridDiameter[2];

	// Seed Particles Randomly according to the grid diameters
	for (int i = 0; i < solverOptions.particle_count; i++)
	{
		h_particles[i].seedParticle(gridDiameter);
	}
	
	// Upload particles to the GPU
	UploadToGPU<Particle<T>>(d_particles, h_particles, sizeof(h_velocityField));

	gpuErrchk(cudaDeviceSynchronize());

	// We do not need particles on CPU anymore;
	delete h_particles;
}




template <typename T>
__host__ bool StreamlineSolver<T>::solve()
{
	InitializeVelocityField();
	InitializeParticles();
	extractStreamlines(this->d_particles,this->d_velocityField);

	
	return true;
}

template <typename T>
 __host__ void StreamlineSolver<T>::extractStreamlines(Particle<T>* d_particles, VelocityField<T>* d_velocityField)
{

	TracingParticles <<< 1, solverOptions.particle_count >>> (d_particles, d_velocityField, solverOptions,reinterpret_cast<Vertex *>(this->p_VertexBuffer));
}


 template <typename T>
__global__ void TracingParticles(Particle<T>* d_particles, VelocityField<T>* d_velocityField, SolverOptions solverOption, Vertex * p_VertexBuffer)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int timesteps = solverOption.timestep;
	float dt = solverOption.dt;

	
	for (int i = 0; i < timesteps; i++)
	{
		d_particles[index].move(dt, d_velocityField);
	}
	p_VertexBuffer[0].pos.x = 0.0;
	p_VertexBuffer[0].pos.y = 0.0;
	p_VertexBuffer[0].pos.z = 0.0;
}
