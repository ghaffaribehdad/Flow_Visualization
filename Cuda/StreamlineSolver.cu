#include "StreamlineSolver.h"
#include "helper_math.h"

#include "texture_fetch_functions.h"

// Kernel of the streamlines, TO-DO: Divide kernel into seprate functions

__global__ void TracingStream(Particle* d_particles, cudaTextureObject_t t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

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

		for (int i = 0; i < lineLength; i++)
		{



			d_particles[index].move(dt, gridDiameter, t_VelocityField);


			// write the new position into the vertex buffer
			switch (solverOptions.projection)
			{
				case Projection::NO_PROJECTION:
					{
						p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
						p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
						p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);
						break;
					}

				case Projection::ZY_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.x = - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);

					break;
				}

				case Projection::XZ_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y =  - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);

					break;
				}

				case Projection::XY_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = - (gridDiameter.z / 2.0);

					break;
				}
				
			
			}



			float3* velocity = d_particles[index].getVelocity();
			float3 norm = normalize(*velocity);
			p_VertexBuffer[index_buffer + i].tangent.x = norm.x;
			p_VertexBuffer[index_buffer + i].tangent.y = norm.y;
			p_VertexBuffer[index_buffer + i].tangent.z = norm.z;
			p_VertexBuffer[index_buffer + i].LineID = index;


			switch (solverOptions.colorMode)
			{
				case 0: // Velocity
				{
				
					p_VertexBuffer[index_buffer + i].measure = VecMagnitude(*velocity);
					break;

				}
				case 1: // Vx
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->x;;
					break;
				}
				case 2: // Vy
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->y;
					break;
				}
				case 3: // Vz
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->z;
					break;
				}
			}
		}
	}
}

__host__ void StreamlineSolver::release()
{
	cudaFree(this->d_Particles);
	cudaFree(this->d_VelocityField);
	this->volumeTexture.release();
}

__host__ bool StreamlineSolver::solve()
{
	// Read Dataset
	this->volume_IO.Initialize(this->solverOptions);
	this->h_VelocityField = InitializeVelocityField(this->solverOptions->currentIdx);
	
	// Copy data to the texture memory
	this->volumeTexture.setField(h_VelocityField);
	this->volumeTexture.setSolverOptions(this->solverOptions);
	this->volumeTexture.initialize();


	// Release it from Host
	volume_IO.release();
	

	this->InitializeParticles(SeedingPattern::SEED_RANDOM);
	
	int blockDim = 256;
	int thread = (this->solverOptions->lines_count / blockDim)+1;
	
	TracingStream << <blockDim , thread >> > (this->d_Particles, volumeTexture.getTexture(), *this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	this->release();

	return true;
}