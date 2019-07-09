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
	cudaDestroyTextureObject(this->t_VelocityField_0);
	cudaDestroyTextureObject(this->t_VelocityField_1);

}



template <typename T>
__host__ bool PathlineSolver<T>::solve()
{
	//At least two timesteps is needed
	int timeSteps = solverOptions.lastIdx - solverOptions.firstIdx;
	if (solverOptions.lastIdx - solverOptions.firstIdx < 2)
		return false; 

	// Initialize Volume IO (Save file path and file names)
	this->volume_IO.Initialize(this->solverOptions);

	// Initialize Particles and upload it to GPU
	this->InitializeParticles();

	int blockDim = 256;
	int thread = (this->solverOptions.lines_count / blockDim) + 1;
	
	solverOptions.lineLength = timeSteps;
	bool odd = true;

	// we go through each time step and solve RK4 for even time steps the first texture is updated,
	// while the second texture is updated for odd time steps
	for (int step = 0; step < timeSteps; step++)
	{
		if (step == 0)
		{
			// For the first timestep we need to load two fields
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx);
			this->InitializeTexture(this->h_VelocityField, this->t_VelocityField_0);
			volume_IO.release();

			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx+1);
			this->InitializeTexture(this->h_VelocityField, this->t_VelocityField_1);
			volume_IO.release();

			odd = false;

		}
		// for the even timesteps we need to reload only one field (tn+1 in the first texture)
		else if (step %2 == 0)
		{
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx + 1);
			cudaDestroyTextureObject(this->t_VelocityField_0);
			this->InitializeTexture(this->h_VelocityField, this->t_VelocityField_0);
			volume_IO.release();

			
		}

		// for the odd timesteps we need to reload only one field (tn+1 in the second texture)
		else if (step % 2 != 0)
		{
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx +1);
			cudaDestroyTextureObject(this->t_VelocityField_1);
			this->InitializeTexture(this->h_VelocityField, this->t_VelocityField_1);
			volume_IO.release();
		}

		TracingPath<T> << <blockDim, thread >> > (this->d_Particles, t_VelocityField_0,t_VelocityField_1, solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer),odd, step);



	}  	
	this->release();
	return true;
}


template <typename T>
__global__ void TracingPath(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;


	if (index < solverOptions.lines_count)
	{
		// needs to be modified
		int index_buffer = index * solverOptions.lineLength;
		float dt = solverOptions.dt;
		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2]
		};


		float3 newPosition = { 0.0f,0.0f,0.0f };

		if (odd)
		{
			if (d_particles[index].isOut())
			{
				newPosition =
				{
					d_particles[index].getPosition()->x,
					d_particles[index].getPosition()->y,
					d_particles[index].getPosition()->z
				};
			}
			else
			{
				newPosition = RK4Odd(t_VelocityField_0, t_VelocityField_1, d_particles[index].getPosition(), gridDiameter, dt);
			}
			
		}
		else //Even
		{
			if (d_particles[index].isOut())
			{
				newPosition =
				{
					d_particles[index].getPosition()->x,
					d_particles[index].getPosition()->y,
					d_particles[index].getPosition()->z
				};
			}
			else
			{
				newPosition = RK4Even(t_VelocityField_0, t_VelocityField_1, d_particles[index].getPosition(), gridDiameter, dt);
			}
		}
		
		d_particles[index].setPosition(newPosition);

		if (!d_particles[index].isOut())
		{
			d_particles[index].checkPosition(gridDiameter);
		}

		// Write into the Vertex BUffer
		p_VertexBuffer[index_buffer + step].pos.x = d_particles[index].getPosition()->x;
		p_VertexBuffer[index_buffer + step].pos.y = d_particles[index].getPosition()->y;
		p_VertexBuffer[index_buffer + step].pos.z = d_particles[index].getPosition()->z;

		switch (solverOptions.colorMode)
		{
			case 0: // Velocity
			{
				float3* velocity = d_particles[index].getVelocity();
				double norm = norm3d(velocity->x, velocity->y, velocity->z);
				p_VertexBuffer[index_buffer + step].colorID.x = norm;
				p_VertexBuffer[index_buffer + step].colorID.y = index;

			}
			case 1: // Vx
			{
				float velocity = d_particles[index].getVelocity()->x;
				p_VertexBuffer[index_buffer + step].colorID.x = velocity;
				p_VertexBuffer[index_buffer + step].colorID.y = index;
			}
			case 2: // Vx
			{
				float velocity = d_particles[index].getVelocity()->y;
				p_VertexBuffer[index_buffer + step].colorID.x = velocity;
				p_VertexBuffer[index_buffer + step].colorID.y = index;
			}
			case 3: // Vx
			{
				float velocity = d_particles[index].getVelocity()->z;
				p_VertexBuffer[index_buffer + step].colorID.x = velocity;
				p_VertexBuffer[index_buffer + step].colorID.y = index;
			}
		}

	}
}
