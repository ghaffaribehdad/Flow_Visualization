#include "StreamlineSolver.cuh"

__host__ bool StreamlineSolver::InitializeCUDA()
{
	// 1. Read and store Vector Field *OK
	h_velocityField = new VelocityField;
	

	std::string fullPath = "";
	fullPath += std::string(this->solverOptions.filePath);
	fullPath += std::string(this->solverOptions.fileName);
	fullPath += std::to_string(this->solverOptions.timestep);
	fullPath += ".bin";

	std::vector<char>* p_vec_buffer = new std::vector<char>;

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

	// 4. Initialize and Upload Particles
	int particleCount = 100;
	this->h_particles = new Particle[particleCount]; // TO-DO: Dynamic size of particle from ImGui
	this->InitializeParticles(particleCount, gridDiameter, SEED_RANDOM);
	this->d_particles = UploadToGPU<Particle>(h_particles, sizeof(h_velocityField));

	// 5. Run The Kernel



	return true;
}

__global__ void TracingParticles(Particle* d_particles, VelocityField* d_velocityField)
{
	int index = threadIdx.x;
}