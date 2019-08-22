#include "Pathlinesolver.h"
#include "CudaHelper.cuh"



void PathlineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);


	this->volumeTexture_0.release();
	this->volumeTexture_1.release();
}



__host__ bool PathlineSolver::solve()
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

	// set solverOptions once
	this->volumeTexture_0.setSolverOptions(&this->solverOptions);
	this->volumeTexture_1.setSolverOptions(&this->solverOptions);


	// we go through each time step and solve RK4 for even time steps the first texture is updated,
	// while the second texture is updated for odd time steps
	for (int step = 0; step < timeSteps; step++)
	{
		if (step == 0)
		{

			// For the first timestep we need to load two fields
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx);
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize();
			volume_IO.release();



			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx+1);
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();
			volume_IO.release();

			odd = false;

		}
		// for the even timesteps we need to reload only one field (tn+1 in the first texture)
		else if (step %2 == 0)
		{
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx + 1);

			this->volumeTexture_0.release();
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx + 1);
			this->volumeTexture_0.setField(h_VelocityField);
			this->volumeTexture_0.initialize();

			volume_IO.release();

			
		}

		// for the odd timesteps we need to reload only one field (tn+1 in the second texture)
		else if (step % 2 != 0)
		{

			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx + 1);

			this->volumeTexture_1.release();
			this->h_VelocityField = this->InitializeVelocityField(solverOptions.firstIdx + 1);
			this->volumeTexture_1.setField(h_VelocityField);
			this->volumeTexture_1.initialize();

			volume_IO.release();

		}

		TracingPath<float> << <blockDim, thread >> > (this->d_Particles, volumeTexture_0.getTexture(), volumeTexture_1.getTexture(), solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer),odd, step);



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
		float3* velocity = d_particles[index_buffer + step].getVelocity();
		float3 norm = normalize(*velocity);
		p_VertexBuffer[index_buffer + step].tangent.x = norm.x;
		p_VertexBuffer[index_buffer + step].tangent.y = norm.y;
		p_VertexBuffer[index_buffer + step].tangent.z = norm.z;
		p_VertexBuffer[index_buffer + step].LineID = float(index) / float(solverOptions.lines_count);

		switch (solverOptions.colorMode)
		{
			case 0: // Velocity
			{
				float3* velocity = d_particles[index].getVelocity();
				p_VertexBuffer[index_buffer + step].measure = norm3df(velocity->x, velocity->y, velocity->z);;
				break;

			}
			case 1: // Vx
			{

				p_VertexBuffer[index_buffer + step].measure = d_particles[index].getVelocity()->x;
				break;
			}
			case 2: // Vy
			{
				float velocity = 
				p_VertexBuffer[index_buffer + step].measure = d_particles[index].getVelocity()->y;
				break;
			}
			case 3: // Vz
			{
				p_VertexBuffer[index_buffer + step].measure = d_particles[index].getVelocity()->z;
				break;

			}
		

		}


	}
}
	
