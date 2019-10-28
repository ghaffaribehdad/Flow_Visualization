#include "CudaHelperFunctions.h"


__device__ float3 RK4Even(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, float3 * position, float3 gridDiameter, float dt)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = {
		position->x  / gridDiameter.x,
		position->y  / gridDiameter.y,
		position->z  / gridDiameter.z
	};
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z};
	k1 = velocity*dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################
	
	float3 k4 = { 0,0,0 };

	relativePos = {
   (position->x + k3.x) / gridDiameter.x,
   (position->y + k3.y) / gridDiameter.y,
   (position->z + k3.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt*velocity;

	float3 newPosition = { 0,0,0 };
	newPosition.x = position->x + (1.0 / 6.0) * (k1.x + 2.0 * k2.x + 2 * k3.x + k4.x);
	newPosition.y = position->y + (1.0 / 6.0) * (k1.y + 2.0 * k2.y + 2 * k3.y + k4.y);
	newPosition.z = position->z + (1.0 / 6.0) * (k1.z + 2.0 * k2.z + 2 * k3.z + k4.z);

	return newPosition;
}


__device__ float3 RK4Odd(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, float3* position, float3 gridDiameter, float dt)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = {
		position->x / gridDiameter.x,
		position->y / gridDiameter.y,
		position->z / gridDiameter.z
	};
	float4 velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
	k1 = velocity * dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = {
   (position->x + k3.x) / gridDiameter.x,
   (position->y + k3.y) / gridDiameter.y,
   (position->z + k3.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity;

	float3 newPosition = { 0,0,0 };
	newPosition.x = position->x + (1.0 / 6.0) * (k1.x + 2.0 * k2.x + 2 * k3.x + k4.x);
	newPosition.y = position->y + (1.0 / 6.0) * (k1.y + 2.0 * k2.y + 2 * k3.y + k4.y);
	newPosition.z = position->z + (1.0 / 6.0) * (k1.z + 2.0 * k2.z + 2 * k3.z + k4.z);

	return newPosition;
}





__device__ float3 RK4Stream(cudaTextureObject_t t_VelocityField_0, float3* position, float3 gridDiameter, float dt)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = {
		position->x / gridDiameter.x,
		position->y / gridDiameter.y,
		position->z / gridDiameter.z
	};
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
	k1 = velocity * dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;

	relativePos = {
	   (position->x + k1.x) / gridDiameter.x,
	   (position->y + k1.y) / gridDiameter.y,
	   (position->z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	relativePos = {
	   (position->x + k2.x) / gridDiameter.x,
	   (position->y + k2.y) / gridDiameter.y,
	   (position->z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = {
   (position->x + k3.x) / gridDiameter.x,
   (position->y + k3.y) / gridDiameter.y,
   (position->z + k3.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity;

	float3 newPosition = { 0,0,0 };
	newPosition.x = position->x + (1.0 / 6.0) * (k1.x + 2.0 * k2.x + 2 * k3.x + k4.x);
	newPosition.y = position->y + (1.0 / 6.0) * (k1.y + 2.0 * k2.y + 2 * k3.y + k4.y);
	newPosition.z = position->z + (1.0 / 6.0) * (k1.z + 2.0 * k2.z + 2 * k3.z + k4.z);

	return newPosition;
}



__device__ void RK4Stream(cudaTextureObject_t t_VelocityField_0, Particle* particle, float3 gridDiameter, float dt)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = {
		particle->m_position.x / gridDiameter.x,
		particle->m_position.y / gridDiameter.y,
		particle->m_position.z / gridDiameter.z
	};
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
	k1 = velocity * dt;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = {
	   (particle->m_position.x + k1.x) / gridDiameter.x,
	   (particle->m_position.y + k1.y) / gridDiameter.y,
	   (particle->m_position.z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;

	relativePos = {
	   (particle->m_position.x + k1.x) / gridDiameter.x,
	   (particle->m_position.y + k1.y) / gridDiameter.y,
	   (particle->m_position.z + k1.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = {
	   (particle->m_position.x + k2.x) / gridDiameter.x,
	   (particle->m_position.y + k2.y) / gridDiameter.y,
	   (particle->m_position.z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	relativePos = {
	   (particle->m_position.x + k2.x) / gridDiameter.x,
	   (particle->m_position.y + k2.y) / gridDiameter.y,
	   (particle->m_position.z + k2.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = {
   (particle->m_position.x + k3.x) / gridDiameter.x,
   (particle->m_position.y + k3.y) / gridDiameter.y,
   (particle->m_position.z + k3.z) / gridDiameter.z
	};
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity;

	float3 newPosition = { 0,0,0 };
	newPosition.x = particle->m_position.x + (1.0 / 6.0) * (k1.x + 2.0 * k2.x + 2 * k3.x + k4.x);
	newPosition.y = particle->m_position.y + (1.0 / 6.0) * (k1.y + 2.0 * k2.y + 2 * k3.y + k4.y);
	newPosition.z = particle->m_position.z + (1.0 / 6.0) * (k1.z + 2.0 * k2.z + 2 * k3.z + k4.z);

	particle->m_position = newPosition;
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
	for (int i = 0; i < solverOptions->lines_count; i++)
	{
		particle[i].m_position.x = solverOptions->gridDiameter[0] / 2.0f -
			solverOptions->seedBox[0] / 2.0f + solverOptions->seedBoxPos[0] +
			static_cast <float> (rand()) /
			static_cast <float> (RAND_MAX /
				solverOptions->seedBox[0]);


		particle[i].m_position.y = solverOptions->gridDiameter[1] / 2.0f -
			solverOptions->seedBox[1] / 2.0f + solverOptions->seedBoxPos[1] +
			static_cast <float> (rand()) /
			static_cast <float> (RAND_MAX / solverOptions->seedBox[1]);


		particle[i].m_position.z = solverOptions->gridDiameter[2] / 2.0f -
			solverOptions->seedBox[2] / 2.0f + solverOptions->seedBoxPos[2] +
			static_cast <float> (rand()) /
			static_cast <float> (RAND_MAX / solverOptions->seedBox[2]);
	}

}



__global__ void TracingPath(Particle* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{
		int line_index = index * solverOptions.lineLength;
		float dt = solverOptions.dt;
		float3 gridDiameter =
		{
			solverOptions.gridDiameter[0],
			solverOptions.gridDiameter[1],
			solverOptions.gridDiameter[2]
		};


		float3 newPosition = { 0.0f,0.0f,0.0f };

		if (d_particles[index].isOut())
		{
			newPosition =
			{
				d_particles[index].getPosition()->x,
				d_particles[index].getPosition()->y,
				d_particles[index].getPosition()->z
			};
		}
		else if (odd)
		{
			newPosition = RK4Odd(t_VelocityField_0, t_VelocityField_1, d_particles[index].getPosition(), gridDiameter, dt);
		}
		else if (!odd) //Even
		{

			newPosition = RK4Even(t_VelocityField_0, t_VelocityField_1, d_particles[index].getPosition(), gridDiameter, dt);
		}

		d_particles[index].setPosition(newPosition);
		d_particles[index].updateVelocity(gridDiameter, t_VelocityField_1);


		if (!d_particles[index].isOut())
		{
			d_particles[index].checkPosition(gridDiameter);
		}

		// Write into the Vertex BUffer
		p_VertexBuffer[line_index + step].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
		p_VertexBuffer[line_index + step].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
		p_VertexBuffer[line_index + step].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);


		float3* velocity = d_particles[index].getVelocity();
		float3 norm = normalize(make_float3(velocity->x, velocity->y, velocity->z));

		p_VertexBuffer[line_index + step].tangent.x = norm.x;
		p_VertexBuffer[line_index + step].tangent.y = norm.y;
		p_VertexBuffer[line_index + step].tangent.z = norm.z;

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