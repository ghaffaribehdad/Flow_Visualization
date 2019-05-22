#include "StreamlineSolver.cuh"

void StreamlineSolver::InitializeVelocityField()
{
	// 1. Read and store Vector Field *OK
	h_velocityField = new VelocityField;
	

	std::string fullPath = "";
	fullPath += std::string(this->solverOptions.filePath);
	fullPath += std::string(this->solverOptions.fileName);
	fullPath += std::to_string(this->solverOptions.timestep);
	fullPath += ".bin";

	std::vector<char>* p_vec_buffer = new std::vector<char>;


	// DEBGU:: path is fixed inside ReadField function
	this->ReadField(p_vec_buffer, fullPath);

	char* p_vec_buffer_temp = &(p_vec_buffer ->at(0));

	this->h_velocityField->setVelocityField(reinterpret_cast<float*> (p_vec_buffer_temp));

	delete p_vec_buffer;

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
	this->d_velocityField = UploadToGPU<VelocityField>(h_velocityField, sizeof(h_velocityField));

}






void StreamlineSolver::InitializeParticles()
{
	// Create an array of particles
	this->h_particles = new Particle[solverOptions.particle_count];

	float3 gridDiameter;
	gridDiameter.x = this->solverOptions.gridDiameter[0];
	gridDiameter.y = this->solverOptions.gridDiameter[1];
	gridDiameter.z = this->solverOptions.gridDiameter[2];

	// Seed Particles Randomly according to the grid diameters
	for (int i = 0; i < solverOptions.particle_count; i++)
	{
		h_particles[i].seedParticle(gridDiameter);
		float3 gridDiametrs = { solverOptions.gridDiameter[0],solverOptions.gridDiameter[1], solverOptions.gridDiameter[2]};
		int3 gridSize = { solverOptions.gridSize[0],solverOptions.gridSize[1], solverOptions.gridSize[2] };

		//h_particles[i].updateVelocity(gridDiametrs, gridSize, h_velocityField);
		OutputDebugStringA("Particles are seeded\n");
	}
	
	// Upload particles to the GPU
	this->d_particles = UploadToGPU<Particle>(h_particles, sizeof(h_velocityField));

	gpuErrchk(cudaDeviceSynchronize());

	// We do not need particles on CPU anymore;
	delete h_particles;
}





__host__ bool StreamlineSolver::solve()
{
	InitializeVelocityField();
	InitializeParticles();
	extractStreamlines();

	
	return true;
}

 __host__ void StreamlineSolver::extractStreamlines()
{

	TracingParticles <<< 1, solverOptions.particle_count >>> (d_particles,  d_velocityField, solverOptions,reinterpret_cast<Vertex *>(this->p_VertexBuffer));
}


__global__ void TracingParticles(Particle* d_particles, VelocityField* d_velocityField, SolverOptions solverOption, Vertex * p_VertexBuffer)
{
	int timesteps = solverOption.timestep;
	float dt = solverOption.dt;
	for (int i = 0; i < timesteps; i++)
	{
		int index = blockDim.x * blockIdx.x + threadIdx.x;

		d_particles[index].move(dt, d_velocityField);
	}
	p_VertexBuffer[0].pos.x = 0.0;
	p_VertexBuffer[0].pos.y = 0.0;
	p_VertexBuffer[0].pos.z = 0.0;
}
