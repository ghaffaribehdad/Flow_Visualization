#include "StreamlineSolver.cuh"

// Explicit instantions
template class StreamlineSolver<float>;
template class StreamlineSolver<double>;

// Kernel of the streamlines, TO-DO: Divide kernel into seprate functions
template <typename T>
__global__ void TracingStream(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{
		int lineLength = solverOptions.lineLength;
		int index_buffer = index * lineLength;
		float dt = solverOptions.dt;
		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2]
		};

		int3 gridSize =
		{
			solverOptions.gridSize[0],
			solverOptions.gridSize[1],
			solverOptions.gridSize[2]
		};

		for (int i = 0; i < lineLength; i++)
		{
			d_particles[index].move(dt, gridSize, gridDiameter, t_VelocityField);

			p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x;
			p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y;
			p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z;

			switch (solverOptions.colorMode)
			{
				case 0: // Velocity
				{
					float3* velocity = d_particles[index].getVelocity();
					double norm = norm3d(velocity->x, velocity->y, velocity->z);
					p_VertexBuffer[index_buffer + i].colorID.x = norm;
					p_VertexBuffer[index_buffer + i].colorID.y = index;

				}
				case 1: // Vx
				{
					float velocity = d_particles[index].getVelocity()->x;
					p_VertexBuffer[index_buffer + i].colorID.x = velocity;
					p_VertexBuffer[index_buffer + i].colorID.y = index;
				}
				case 2: // Vx
				{
					float velocity = d_particles[index].getVelocity()->y;
					p_VertexBuffer[index_buffer + i].colorID.x = velocity;
					p_VertexBuffer[index_buffer + i].colorID.y = index;
				}
				case 3: // Vx
				{
					float velocity = d_particles[index].getVelocity()->z;
					p_VertexBuffer[index_buffer + i].colorID.x = velocity;
					p_VertexBuffer[index_buffer + i].colorID.y = index;
				}
			}

			
		}
	}

}
template <typename T>
__host__ void StreamlineSolver<T>::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);
	cudaDestroyTextureObject(this->t_VelocityField);
}

template <typename T>
__host__ bool StreamlineSolver<T>::solve()
{
	this->volume_IO.Initialize(this->solverOptions);

	// TO-DO: Define streamlinesolver for double precision
	this->h_VelocityField = InitializeVelocityField(this->solverOptions.currentIdx);

	this->InitializeTexture(h_VelocityField, t_VelocityField);

	this->InitializeParticles();
	
	int blockDim = 256;
	int thread = (this->solverOptions.lines_count / blockDim)+1;
	
	TracingStream<T> << <blockDim , thread >> > (this->d_Particles, t_VelocityField, solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	this->release();

	return true;
}