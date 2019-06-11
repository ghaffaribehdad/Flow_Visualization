#include "Pathlinesolver.cuh"
#include "CudaHelper.cuh"

// Explicit instantions
template class PathlineSolver<float>;
template class PathlineSolver<double>;


template <typename T>
void PathlineSolver<T>::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);
	cudaDestroyTextureObject(this->t_VelocityField[0]);
	cudaDestroyTextureObject(this->t_VelocityField[1]);
	cudaDestroyTextureObject(this->t_VelocityField[2]);
}

//template <typename T>
//__global__ void TracingParticles(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
//{
//	int index = blockDim.x * blockIdx.x + threadIdx.x;
//	if (index < solverOptions.lineLength)
//	{
//		int lineLength = solverOptions.lineLength;
//		int index_buffer = index * lineLength;
//		float dt = solverOptions.dt;
//		float3 gridDiameter =
//		{
//			solverOptions.gridDiameter[0],
//			solverOptions.gridDiameter[1],
//			solverOptions.gridDiameter[2]
//		};
//
//		int3 gridSize =
//		{
//			solverOptions.gridSize[0],
//			solverOptions.gridSize[1],
//			solverOptions.gridSize[2]
//		};
//
//		for (int i = 0; i < lineLength; i++)
//		{
//			d_particles[index].move(dt, gridSize, gridDiameter, t_VelocityField);
//
//			p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x;
//			p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y;
//			p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z;
//
//			switch (solverOptions.colorMode)
//			{
//			case 0: // Velocity
//			{
//				float3* velocity = d_particles[index].getVelocity();
//				double norm = norm3d(velocity->x, velocity->y, velocity->z);
//				p_VertexBuffer[index_buffer + i].colorID.x = norm;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//
//			}
//			case 1: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->x;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			case 2: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->y;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			case 3: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->z;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			}
//
//
//		}
//	}
//
//}

template <typename T>
__host__ bool PathlineSolver<T>::solve()
{
	if (solverOptions.lastIdx - solverOptions.firstIdx < 3)
		return false; //At least three timestep is needed

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.Initialize(this->solverOptions);

	// Initialize Particles and upload it to GPU
	this->InitializeParticles();

	int blockDim = 256;
	int thread = (this->solverOptions.lines_count / blockDim) + 1;

	for (int timeStep = solverOptions.firstIdx; timeStep <= solverOptions.lastIdx; timeStep++)
	{
		// For the very first step we need to initialize all three textures
		if (timeStep == 0)
		{
			for (int i = 0; i++; i < 3)
			{
				this->h_VelocityField = this->InitializeVelocityField(timeStep + i);
				this->InitializeTexture(this->h_VelocityField, this->t_VelocityField[i]);

			}
			//Run the kernel for the first step
		}

		else // for the rest only one of the textures has to be updated
		{
			//update texture
			//Run kernel
		}

		this->release();
	}  	

	return true;
}


template <typename T>
__global__ void TracingPath(Particle<T>* d_particles, cudaTextureObject_t* t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{
		int lineLength = solverOptions.lineLength;		//
		int index_buffer = index * lineLength;			//
		float dt = solverOptions.dt;					//
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
