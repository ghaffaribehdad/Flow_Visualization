#include "StreamlineSolver.h"
#include "helper_math.h"
#include "..//Cuda/CudaHelperFunctions.h"
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

		float3 temp_position = *d_particles[index].getPosition();

		d_particles[index].updateVelocity(gridDiameter, t_VelocityField);
		

		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);

		if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(1.0f, 0.0f, 0.0f);

		if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(0.0f, 0.0f, 1.0f);



		for (int i = 0; i < lineLength; i++)
		{
			if (solverOptions.periodic)
			{
				p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
				p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
				p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);
			}
			else 
			{
				if (!d_particles[index].isOut())
				{
					d_particles[index].checkPosition(gridDiameter);
				}

				if (d_particles[index].isOut() && i != 0)
				{
					p_VertexBuffer[index_buffer + i].pos.x = p_VertexBuffer[index_buffer + i - 1].pos.x;
					p_VertexBuffer[index_buffer + i].pos.y = p_VertexBuffer[index_buffer + i - 1].pos.y;
					p_VertexBuffer[index_buffer + i].pos.z = p_VertexBuffer[index_buffer + i - 1].pos.z;
				}
				else
				{
					p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);

				}
			}



			float3* velocity = d_particles[index].getVelocity();
			float3 tangent = normalize(*velocity);
			


			p_VertexBuffer[index_buffer + i].normal.x = upDir.x;
			p_VertexBuffer[index_buffer + i].normal.y = upDir.y;
			p_VertexBuffer[index_buffer + i].normal.z = upDir.z;

			p_VertexBuffer[index_buffer + i].tangent.x = tangent.x;
			p_VertexBuffer[index_buffer + i].tangent.y = tangent.y;
			p_VertexBuffer[index_buffer + i].tangent.z = tangent.z;


			
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

			// Do not check if it is out
			RK4Stream(t_VelocityField, &d_particles[index], gridDiameter, dt);

			// Update position based on the projection
			switch (solverOptions.projection)
			{
				case Projection::NO_PROJECTION:
				{
					break;
				}
				case Projection::ZY_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.x = temp_position.x - (gridDiameter.x / 2.0);
	

					break;
				}
				case Projection::XZ_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.y = temp_position.y - (gridDiameter.y / 2.0);


					break;
				}
				case Projection::XY_PROJECTION:
				{

					p_VertexBuffer[index_buffer + i].pos.z = temp_position.z - (gridDiameter.z / 2.0);

					break;
				}
			}




		}//end of for loop
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
	

	this->InitializeParticles(static_cast<SeedingPattern>( this->solverOptions->seedingPattern));

	int blockDim = 1024;
	int thread = (this->solverOptions->lines_count / blockDim)+1;
	
	TracingStream << <blockDim , thread >> > (this->d_Particles, volumeTexture.getTexture(), *this->solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	this->release();

	return true;
}