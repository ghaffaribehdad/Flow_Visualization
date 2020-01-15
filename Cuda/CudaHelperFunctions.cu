#include "CudaHelperFunctions.h"
#include <random>

__device__ void RK4Path
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	float dt,
	bool periodicity
)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };

	float3 relativePos = particle->m_position / gridDiameter;

	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
	k1 = velocity * dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = (particle->m_position + k1) / gridDiameter;

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;


	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = (particle->m_position + k2) / gridDiameter;

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = (particle->m_position + k3) / gridDiameter;
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity;

	if (periodicity)
	{
		particle->m_position = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);
		particle->updateVelocity(gridDiameter, t_VelocityField_1);
	}
	else
	{
		if (particle->m_position < gridDiameter && particle->m_position > make_float3(0.0f, 0.0f, 0.0f))
		{
			particle->m_position = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);
			particle->updateVelocity(gridDiameter, t_VelocityField_1);
		}
		else
		{
			particle->outOfScope = true;
		}
	}


}








__device__ void RK4Stream(cudaTextureObject_t t_VelocityField_0, Particle* particle, float3 gridDiameter, float dt)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = particle->m_position / gridDiameter; 
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);

	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k1 = velocity * dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos =    (particle->m_position + k1) / gridDiameter;
	 
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;


	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = (particle->m_position + k2) / gridDiameter;

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;


	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = (particle->m_position + k3) / gridDiameter;

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity;

	particle->m_position = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);

	particle->updateVelocity(gridDiameter, t_VelocityField_0);
	
}


__host__ void seedParticleGridPoints(Particle* particle, const SolverOptions* solverOptions)
{
	float3 gridMeshSize =
	{
		solverOptions->seedBox[0] / (float)solverOptions->seedGrid[0],
		solverOptions->seedBox[1] / (float)solverOptions->seedGrid[1],
		solverOptions->seedBox[2] / (float)solverOptions->seedGrid[2],
	};

	for (int x = 0; x < solverOptions->seedGrid[0]; x++)
	{
		for (int y = 0; y < solverOptions->seedGrid[1]; y++)
		{
			for (int z = 0; z < solverOptions->seedGrid[2]; z++)
			{
				int index = x * solverOptions->seedGrid[1] * solverOptions->seedGrid[2] + y * solverOptions->seedGrid[2] + z;
				particle[index].m_position = 
				{
					
					solverOptions->gridDiameter[0] / 2.0f -
					solverOptions->seedBox[0] / 2.0f +
					solverOptions->seedBoxPos[0] + (float)x * gridMeshSize.x,

					solverOptions->gridDiameter[1] / 2.0f -
					solverOptions->seedBox[1] / 2.0f +
					solverOptions->seedBoxPos[1]  + (float)y * gridMeshSize.y,

					solverOptions->gridDiameter[2] / 2.0f -
					solverOptions->seedBox[2] / 2.0f +
					solverOptions->seedBoxPos[2]  + (float)z * gridMeshSize.z
				};

			}
		}
	}





}

__host__ void seedParticleRandom(Particle * particle, const SolverOptions * solverOptions)
{

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	generator.seed(solverOptions->counter);

	for (int i = 0; i < solverOptions->lines_count; i++)
	{
		particle[i].m_position.x = solverOptions->gridDiameter[0] / 2.0f -
			solverOptions->seedBox[0] / 2.0f + solverOptions->seedBoxPos[0] +
			distribution(generator) * solverOptions->seedBox[0];

		distribution.reset();

		particle[i].m_position.y = solverOptions->gridDiameter[1] / 2.0f -
			solverOptions->seedBox[1] / 2.0f + solverOptions->seedBoxPos[1] +
			distribution(generator) * solverOptions->seedBox[1];

		distribution.reset();

		particle[i].m_position.z = solverOptions->gridDiameter[2] / 2.0f -
			solverOptions->seedBox[2] / 2.0f + solverOptions->seedBoxPos[2] +
			distribution(generator) * solverOptions->seedBox[2];
		distribution.reset();

	}


}



__global__ void TracingPath(Particle* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{

		if (!d_particles[index].outOfScope)
		{
			// line_index indicates the line segment index
			int line_index = index * solverOptions.lineLength;

			float3 gridDiameter = 
			{
				solverOptions.gridDiameter[0],
				solverOptions.gridDiameter[1],
				solverOptions.gridDiameter[2]
			};


			if (odd)
			{
				RK4Path(t_VelocityField_1, t_VelocityField_0, &d_particles[index], gridDiameter, solverOptions.dt,false);
			}
			else //Even
			{

				RK4Path(t_VelocityField_0, t_VelocityField_1, &d_particles[index], gridDiameter, solverOptions.dt,false);
			}

			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);

			// check if the up vector is parallel to the tangent
			if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
				upDir = make_float3(1.0f, 0.0f, 0.0f); // if yes switch the up vector

			if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
				upDir = make_float3(0.0f, 0.0f, 1.0f);


			// Write into the Vertex BUffer
			p_VertexBuffer[line_index + step].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
			p_VertexBuffer[line_index + step].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
			p_VertexBuffer[line_index + step].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


			float3* velocity = d_particles[index].getVelocity();
			float3 tangent = normalize(*velocity);


			p_VertexBuffer[line_index + step].normal.x = upDir.x;
			p_VertexBuffer[line_index + step].normal.y = upDir.y;
			p_VertexBuffer[line_index + step].normal.z = upDir.z;


			p_VertexBuffer[line_index + step].tangent.x = tangent.x;
			p_VertexBuffer[line_index + step].tangent.y = tangent.y;
			p_VertexBuffer[line_index + step].tangent.z = tangent.z;

			p_VertexBuffer[line_index + step].LineID = index;

			switch (solverOptions.colorMode)
			{
				case 0: // Velocity
				{
					p_VertexBuffer[line_index + step].measure = VecMagnitude(*velocity);
					break;

				}
				case 1: // Vx
				{

					p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->x;
					break;
				}
				case 2: // Vy
				{
					p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->y;
					break;
				}
				case 3: // Vz
				{
					p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->z;
					break;

				}
			}
		}		
	}
}


__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
)
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

		float3 initialVelocity = d_particles[index].m_velocity;

		float3 upDir = make_float3(0.0f, 0.0f, 1.0f);

		if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(1.0f, 0.0f, 0.0f);

		else if (abs(dot(upDir, normalize(d_particles[index].m_velocity)))> 0.1f)
			upDir = make_float3(0.0f, 1.0f, 0.0f);



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
				case 4: // initial Vx
				{
					p_VertexBuffer[index_buffer + i].measure = initialVelocity.x;
					break;
				}
				case 5: // initial Vy
				{
					p_VertexBuffer[index_buffer + i].measure = initialVelocity.y;
					break;
				}
				case 6: // initial Vz
				{
					p_VertexBuffer[index_buffer + i].measure = initialVelocity.z;
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


__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer,
	float4 * d_VertexBuffer
)
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



		float3 upDir = make_float3(0.0f, 0.0f, 1.0f);

		if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(1.0f, 0.0f, 0.0f);

		else if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(0.0f, 1.0f, 0.0f);


		float3 relativePos = d_particles[index].m_position / gridDiameter;
		float4 velocity4D_initial = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z);


		for (int i = 0; i < lineLength; i++)
		{
			if (solverOptions.periodic)
			{
				p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
				p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
				p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


				// Write into d_VertexBuffer
				d_VertexBuffer[index_buffer + i].x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
				d_VertexBuffer[index_buffer + i].y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
				d_VertexBuffer[index_buffer + i].z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);
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

					d_VertexBuffer[index_buffer + i].x = p_VertexBuffer[index_buffer + i - 1].pos.x;
					d_VertexBuffer[index_buffer + i].y = p_VertexBuffer[index_buffer + i - 1].pos.y;
					d_VertexBuffer[index_buffer + i].z = p_VertexBuffer[index_buffer + i - 1].pos.z;
				}
				else
				{
					p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


					d_VertexBuffer[index_buffer + i].x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					d_VertexBuffer[index_buffer + i].y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					d_VertexBuffer[index_buffer + i].z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


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

					d_VertexBuffer[index_buffer + i].w = VecMagnitude(*velocity);

					break;

				}
				case 1: // Vx
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->x;

					d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->x;

					break;
				}
				case 2: // Vy
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->y;

					d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->y;
					break;
				}
				case 3: // Vz
				{
					p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->z;

					d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->z;
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



__global__ void TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	cudaTextureObject_t t_Vorticity,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer,
	float4* d_VertexBuffer
)
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



		float3 upDir = make_float3(0.0f, 0.0f, 1.0f);

		if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(1.0f, 0.0f, 0.0f);

		else if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.1f)
			upDir = make_float3(0.0f, 1.0f, 0.0f);



		for (int i = 0; i < lineLength; i++)
		{
			if (solverOptions.periodic)
			{
				p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
				p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
				p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


				// Write into d_VertexBuffer
				d_VertexBuffer[index_buffer + i].x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
				d_VertexBuffer[index_buffer + i].y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
				d_VertexBuffer[index_buffer + i].z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);
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

					d_VertexBuffer[index_buffer + i].x = p_VertexBuffer[index_buffer + i - 1].pos.x;
					d_VertexBuffer[index_buffer + i].y = p_VertexBuffer[index_buffer + i - 1].pos.y;
					d_VertexBuffer[index_buffer + i].z = p_VertexBuffer[index_buffer + i - 1].pos.z;
				}
				else
				{
					p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


					d_VertexBuffer[index_buffer + i].x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
					d_VertexBuffer[index_buffer + i].y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
					d_VertexBuffer[index_buffer + i].z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


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

				float3  relativePos = *d_particles[index].getPosition();
				relativePos = relativePos / make_float3(solverOptions.gridDiameter[0], solverOptions.gridDiameter[1], solverOptions.gridDiameter[2]);
				p_VertexBuffer[index_buffer + i].measure = tex3D<float4>(t_Vorticity, relativePos.x, relativePos.y, relativePos.z).x;

				d_VertexBuffer[index_buffer + i].w = tex3D<float4>(t_Vorticity, relativePos.x, relativePos.y, relativePos.z).x;

				break;

			}
			case 1: // Vx
			{
				p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->x;

				d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->x;

				break;
			}
			case 2: // Vy
			{
				p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->y;

				d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->y;
				break;
			}
			case 3: // Vz
			{
				p_VertexBuffer[index_buffer + i].measure = d_particles[index].getVelocity()->z;

				d_VertexBuffer[index_buffer + i].w = d_particles[index].getVelocity()->z;
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



__device__ float3 binarySearch_heightField
(
	float3 _position,
	cudaSurfaceObject_t tex,
	float3 _samplingStep,
	float3 gridDiameter,
	float tolerance,
	int maxIteration

)
{
	float3 position = _position;
	float3 relative_position = position / gridDiameter;

	float3 samplingStep = _samplingStep;

	bool side = 0; // 0 -> insiede 1-> outside

	int counter = 0;


	while (fabsf(tex2D<float4>(tex, relative_position.x, relative_position.y).x - position.y) > tolerance && counter < maxIteration)
	{

		if ( tex2D<float4>(tex,  relative_position.x, relative_position.y).x - position.y >  0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			// return position if we are out of texture
			if (outofTexture((position - samplingStep) / gridDiameter))
				return position;

			position = position - samplingStep;
			relative_position = position / gridDiameter;

			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			// return position if we are out of texture
			if (outofTexture((position + samplingStep) / gridDiameter))
				return position;

			position = position + samplingStep;
			relative_position = position / gridDiameter;
			side = 1;

		}
		counter++;

	}

	return position;
};



__global__ void Vorticity
(
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	cudaSurfaceObject_t	s_measure
)
{
	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;


	if (index < solverOptions.gridSize[0])
	{


		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2]
		};

		float3 gridSize =
		{
			(float)solverOptions.gridSize[0],
			(float)solverOptions.gridSize[1],
			(float)solverOptions.gridSize[2]

		};

		float3 relativePos = { 0,0,0 };
		float3 h = gridDiameter / make_float3
		(
			solverOptions.gridSize[0],
			solverOptions.gridSize[1],
			solverOptions.gridSize[2]
		);

		float4	dVx = { 0,0,0,0 };
		float4	dVy = { 0, 0, 0,0 };
		float4  dVz = { 0, 0, 0 ,0 };



		for (int i = 0; i < solverOptions.gridSize[1]; i++)
		{
			for (int j = 0; j < solverOptions.gridSize[2]; j++)
			{
				relativePos = make_float3((float)index, (float)i, (float)j);
				relativePos = relativePos / gridSize;


				dVx = tex3D<float4>(t_VelocityField, relativePos.x + h.x / 2.0, relativePos.y, relativePos.z);
				dVx -= tex3D<float4>(t_VelocityField, relativePos.x - h.x / 2.0, relativePos.y, relativePos.z);

				dVy = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y + h.x / 2.0, relativePos.z);
				dVy -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y - h.x / 2.0, relativePos.z);

				dVz = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z + h.x / 2.0);
				dVz -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z - h.x / 2.0);

				// This would give us the Jacobian Matrix
				dVx = dVx / h.x;
				dVy = dVy / h.y;
				dVz = dVz / h.z;

				// Calculate the curl (vorticity vector)
				float3 vorticity_vec = make_float3
				(
					dVy.z - dVz.y,
					dVz.x - dVx.z,
					dVx.y - dVy.x
				);
				// calculate the vorticity magnitude
				float vorticity_mag = sqrtf(dot(vorticity_vec, vorticity_vec));

				float4 value = { vorticity_mag,0,0,0 };
				// Now write it back to the CUDA Surface

				surf3Dwrite(value, s_measure, sizeof(float4) * index, i, j);

			}
		}
	}
	
}