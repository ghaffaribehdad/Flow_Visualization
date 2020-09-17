#include "CudaHelperFunctions.h"
#include <random>
#include "helper_math.h"



__device__ void RK4Stream(
	cudaTextureObject_t t_VelocityField_0,
	Particle* particle,
	const float3& gridDiameter,
	const int3& gridSize,
	float dt,
	float3 velocityScale
)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = world2Tex(particle->m_position, gridDiameter,gridSize);
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z) ;

	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k1 = velocity * dt * velocityScale;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos =   world2Tex(particle->m_position + (k1 * 0.5f), gridDiameter,gridSize);
	 
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	k2 = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = dt * k2 * velocityScale;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = world2Tex(particle->m_position + (k2* 0.5f), gridDiameter, gridSize);

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	k3 = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = dt * k3 * velocityScale;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = world2Tex(particle->m_position + k3, gridDiameter, gridSize);

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity * velocityScale;

	particle->m_position = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);

	particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_0);
	
}







__global__ void TracingStreak(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;


	if (index < solverOptions.lines_count)
	{
		// line_index indicates the line segment index
		int line_index = index * solverOptions.lineLength;
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);


		for (int i = 0; i <= step; i++)
		{
			Particle tempPar;
			if (odd)
			{
				float3 oldPos = make_float3(p_VertexBuffer[line_index + i].pos.x, p_VertexBuffer[line_index + i].pos.y, p_VertexBuffer[line_index + i].pos.z);
				tempPar = RK4Streak(t_VelocityField_1, t_VelocityField_0, oldPos, gridDiameter, gridSize, solverOptions.dt, solverOptions.periodic, Array2Float3(solverOptions.velocityScalingFactor));
				p_VertexBuffer[line_index + i].pos.x = tempPar.m_position.x;
				p_VertexBuffer[line_index + i].pos.y = tempPar.m_position.y;
				p_VertexBuffer[line_index + i].pos.z = tempPar.m_position.z;
			}
			else //Even
			{

				float3 oldPos = make_float3(p_VertexBuffer[line_index + i].pos.x, p_VertexBuffer[line_index + i].pos.y, p_VertexBuffer[line_index + i].pos.z);
				tempPar = RK4Streak(t_VelocityField_0, t_VelocityField_1, oldPos, gridDiameter, gridSize, solverOptions.dt, solverOptions.periodic, Array2Float3(solverOptions.velocityScalingFactor));
				p_VertexBuffer[line_index + i].pos.x = tempPar.m_position.x;
				p_VertexBuffer[line_index + i].pos.y = tempPar.m_position.y;
				p_VertexBuffer[line_index + i].pos.z = tempPar.m_position.z;
			}


			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
			float3 tangent = normalize(tempPar.m_velocity);


			// Update position based on the projection
			if (step != 0)
			{
				switch (solverOptions.projection)
				{
				case Projection::NO_PROJECTION:
				{
					break;
				}
				case Projection::ZY_PROJECTION:
				{
					tangent = normalize(make_float3(0.0f, tempPar.m_velocity.y, tempPar.m_velocity.z));
					break;
				}
				case Projection::XZ_PROJECTION:
				{
					tangent = normalize(make_float3(tempPar.m_velocity.x, 0.0f, tempPar.m_velocity.z));
					break;
				}
				case Projection::XY_PROJECTION:
				{

					tangent = normalize(make_float3(tempPar.m_velocity.x, tempPar.m_velocity.y, 0.0f));
					break;
				}
				}

			}


			// Update position based on the projection
			if (step != 0)
			{
				switch (solverOptions.projection)
				{
				case Projection::NO_PROJECTION:
				{
					break;
				}
				case Projection::ZY_PROJECTION:
				{
					p_VertexBuffer[line_index + i].pos.x = p_VertexBuffer[line_index].pos.x;
					break;
				}
				case Projection::XZ_PROJECTION:
				{
					p_VertexBuffer[line_index + i].pos.y = p_VertexBuffer[line_index].pos.y;
					break;
				}
				case Projection::XY_PROJECTION:
				{

					p_VertexBuffer[line_index + i].pos.z = p_VertexBuffer[line_index].pos.z;
					break;
				}
				}

			}



			p_VertexBuffer[line_index + i].normal.x = upDir.x;
			p_VertexBuffer[line_index + i].normal.y = upDir.y;
			p_VertexBuffer[line_index + i].normal.z = upDir.z;


			p_VertexBuffer[line_index + i].tangent.x = tangent.x;
			p_VertexBuffer[line_index + i].tangent.y = tangent.y;
			p_VertexBuffer[line_index + i].tangent.z = tangent.z;


			////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/////////////
			p_VertexBuffer[line_index + i].LineID = index;

			switch (solverOptions.colorMode)
			{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{
				p_VertexBuffer[line_index + i].measure = magnitude(tempPar.m_velocity);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY: // Vx
			{

				p_VertexBuffer[line_index + i].measure = tempPar.getVelocity()->x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY: // Vy
			{
				p_VertexBuffer[line_index + i].measure = tempPar.getVelocity()->y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY: // Vz
			{
				p_VertexBuffer[line_index + i].measure = tempPar.getVelocity()->z;
				break;

			}

			}
		}
		}



}


__global__ void InitializeVertexBufferStreaklines
(
	Particle* d_particles,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
	)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{
		int line_index = index * solverOptions.lineLength;
		// Write into the Vertex BUffer
		for (int i = 0; i < solverOptions.lineLength; i++)
		{
			p_VertexBuffer[line_index + i].pos.x = d_particles[index].getPosition()->x;
			p_VertexBuffer[line_index + i].pos.y = d_particles[index].getPosition()->y;
			p_VertexBuffer[line_index + i].pos.z = d_particles[index].getPosition()->z;
		}

	}
}


__global__ void AddOffsetVertexBufferStreaklines
(
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < solverOptions.lines_count)
	{
		int line_index = index * solverOptions.lineLength;
		// Write into the Vertex BUffer
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		for (int i = 0; i < solverOptions.lineLength; i++)
		{
			p_VertexBuffer[line_index + i].pos.x = p_VertexBuffer[line_index + i].pos.x - (gridDiameter.x / 2.0);
			p_VertexBuffer[line_index + i].pos.y = p_VertexBuffer[line_index + i].pos.y - (gridDiameter.y / 2.0);
			p_VertexBuffer[line_index + i].pos.z = p_VertexBuffer[line_index + i].pos.z - (gridDiameter.z / 2.0);
		}

	}

}

__global__ void TracingPath(Particle* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;


	if (index < solverOptions.lines_count)
	{
		
		// line_index indicates the line segment index
		int line_index = index * solverOptions.lineLength;

		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);

		float3 current_vel = d_particles[index].m_velocity;
		

		if (odd)
		{
			RK4Path(t_VelocityField_1, t_VelocityField_0, &d_particles[index], gridDiameter, gridSize, solverOptions.dt, solverOptions.periodic,Array2Float3(solverOptions.velocityScalingFactor));
		}
		else //Even
		{

			RK4Path(t_VelocityField_0, t_VelocityField_1, &d_particles[index], gridDiameter, gridSize, solverOptions.dt, solverOptions.periodic, Array2Float3(solverOptions.velocityScalingFactor));
		}




		// use the up vector as normal
		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);


		float3 tangent = normalize(d_particles[index].m_velocity);


		// Update position based on the projection
		if (step != 0)
		{
			switch (solverOptions.projection)
			{
			case Projection::NO_PROJECTION:
			{
				break;
			}
			case Projection::ZY_PROJECTION:
			{
				tangent = normalize(make_float3(0.0f, d_particles[index].m_velocity.y, d_particles[index].m_velocity.z));
				break;
			}
			case Projection::XZ_PROJECTION:
			{
				tangent = normalize(make_float3(d_particles[index].m_velocity.x, 0.0f, d_particles[index].m_velocity.z));
				break;
			}
			case Projection::XY_PROJECTION:
			{

				tangent = normalize(make_float3(d_particles[index].m_velocity.x, d_particles[index].m_velocity.y, 0.0f));
				break;
			}
			}

		}

		// Write into the Vertex BUffer
		p_VertexBuffer[line_index + step].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
		p_VertexBuffer[line_index + step].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
		p_VertexBuffer[line_index + step].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);

	
		// Update position based on the projection
		if (step != 0)
		{
			switch (solverOptions.projection)
			{
			case Projection::NO_PROJECTION:
			{
				break;
			}
			case Projection::ZY_PROJECTION:
			{
				p_VertexBuffer[line_index + step].pos.x = p_VertexBuffer[line_index].pos.x;
				break;
			}
			case Projection::XZ_PROJECTION:
			{
				p_VertexBuffer[line_index + step].pos.y = p_VertexBuffer[line_index].pos.y;
				break;
			}
			case Projection::XY_PROJECTION:
			{

				p_VertexBuffer[line_index + step].pos.z = p_VertexBuffer[line_index].pos.z;
				break;
			}
			}

		}



		p_VertexBuffer[line_index + step].normal.x = upDir.x;
		p_VertexBuffer[line_index + step].normal.y = upDir.y;
		p_VertexBuffer[line_index + step].normal.z = upDir.z;


		p_VertexBuffer[line_index + step].tangent.x = tangent.x;
		p_VertexBuffer[line_index + step].tangent.y = tangent.y;
		p_VertexBuffer[line_index + step].tangent.z = tangent.z;

		p_VertexBuffer[line_index + step].LineID = index;

		switch (solverOptions.colorMode)
		{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{
				p_VertexBuffer[line_index + step].measure = magnitude(d_particles[index].m_velocity);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY: // Vx
			{

				p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY: // Vy
			{
				p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY: // Vz
			{
				p_VertexBuffer[line_index + step].measure = d_particles[index].getVelocity()->z;
				break;

			}
			case ColorMode::ColorMode::CURVATURE:
			{
				float3 gamma1 = current_vel;
				float3 gamma2 = (d_particles[index].m_velocity - gamma1) / Array2Float3(solverOptions.velocityScalingFactor);
				float3 gamma1Xgamma2 = cross(gamma1, gamma2);
				float curvature = magnitude(gamma1Xgamma2);
				curvature = curvature / powf(magnitude(gamma1), 3);
				p_VertexBuffer[line_index + step].measure = curvature;

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
		int index_buffer = index * solverOptions.lineLength;
		float dt = solverOptions.dt;


		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridsize = Array2Int3(solverOptions.gridSize);


		// Update the velocity of the particle at its initial position
		d_particles[index].updateVelocity(gridDiameter, gridsize, t_VelocityField);

		float3 init_pos = *d_particles[index].getPosition();


		// Up direction is needed for Shading (Must be revised)
		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
		//if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.01f)
		//	upDir = make_float3(1.0f, 0.0f, 0.0f);
		//else if (abs(dot(upDir, normalize(d_particles[index].m_velocity))) > 0.01f)
		//	upDir = make_float3(0.0f, 1.0f, 0.0f);



		for (int i = 0; i < solverOptions.lineLength; i++)
		{

			// Center the position of the particles
			p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x - (gridDiameter.x / 2.0);
			p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y - (gridDiameter.y / 2.0);
			p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z - (gridDiameter.z / 2.0);

			// If not periodic, terminate the calculation if the particle is outside of the volume
			if (!solverOptions.periodic)
			{
				if (d_particles[index].isOut())
				{
					p_VertexBuffer[index_buffer + i].pos.x = p_VertexBuffer[index_buffer + i - 1].pos.x;
					p_VertexBuffer[index_buffer + i].pos.y = p_VertexBuffer[index_buffer + i - 1].pos.y;
					p_VertexBuffer[index_buffer + i].pos.z = p_VertexBuffer[index_buffer + i - 1].pos.z;
					continue;
				}
				else
				{
					d_particles[index].checkPosition(gridDiameter);
				}
			}

				float3 current_vel = d_particles[index].m_velocity;


				float3 tangent = normalize(current_vel);

				p_VertexBuffer[index_buffer + i].normal.x = upDir.x;
				p_VertexBuffer[index_buffer + i].normal.y = upDir.y;
				p_VertexBuffer[index_buffer + i].normal.z = upDir.z;

				p_VertexBuffer[index_buffer + i].tangent.x = tangent.x;
				p_VertexBuffer[index_buffer + i].tangent.y = tangent.y;
				p_VertexBuffer[index_buffer + i].tangent.z = tangent.z;

				p_VertexBuffer[index_buffer + i].LineID = index;



				//updates velocity and position of the particle 
				RK4Stream(t_VelocityField, &d_particles[index], gridDiameter, Array2Int3(solverOptions.gridSize), dt, Array2Float3(solverOptions.velocityScalingFactor));




				switch (solverOptions.colorMode)
				{
				case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
				{

					p_VertexBuffer[index_buffer + i].measure = magnitude(current_vel);
					break;

				}
				case ColorMode::ColorMode::U_VELOCITY:
				{
					p_VertexBuffer[index_buffer + i].measure = current_vel.x;
					break;
				}
				case ColorMode::ColorMode::V_VELOCITY:
				{
					p_VertexBuffer[index_buffer + i].measure = current_vel.y;
					break;
				}
				case ColorMode::ColorMode::W_VELOCITY:
				{
					p_VertexBuffer[index_buffer + i].measure = current_vel.z;
					break;
				}
				case ColorMode::ColorMode::CURVATURE:
				{
					float3 gamma1 = current_vel;
					float3 gamma2 = (d_particles[index].m_velocity - gamma1)/dt;
					float3 gamma1Xgamma2 = cross(gamma1, gamma2);
					float curvature = magnitude(gamma1Xgamma2);
					curvature = curvature / powf(magnitude(gamma1), 3);
					p_VertexBuffer[index_buffer + i].measure = curvature;

					
				}

				}

				// Update position based on the projection
				switch (solverOptions.projection)
				{
				case Projection::NO_PROJECTION:
				{
					break;
				}
				case Projection::ZY_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.x = init_pos.x - (gridDiameter.x / 2.0);
					break;
				}
				case Projection::XZ_PROJECTION:
				{
					p_VertexBuffer[index_buffer + i].pos.y = init_pos.y - (gridDiameter.y / 2.0);
					break;
				}
				case Projection::XY_PROJECTION:
				{

					p_VertexBuffer[index_buffer + i].pos.z = init_pos.z - (gridDiameter.z / 2.0);
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



//__global__ void Vorticity
//(
//	cudaTextureObject_t t_VelocityField,
//	SolverOptions solverOptions,
//	cudaSurfaceObject_t	s_measure
//)
//{
//	int index = blockIdx.x * blockDim.y * blockDim.x;
//	index += threadIdx.y * blockDim.x;
//	index += threadIdx.x;
//
//
//	if (index < solverOptions.gridSize[0])
//	{
//
//
//		float3 gridDiameter =
//		{
//			solverOptions.gridDiameter[0],
//			solverOptions.gridDiameter[1],
//			solverOptions.gridDiameter[2]
//		};
//
//		float3 gridSize =
//		{
//			(float)solverOptions.gridSize[0],
//			(float)solverOptions.gridSize[1],
//			(float)solverOptions.gridSize[2]
//
//		};
//
//		float3 relativePos = { 0,0,0 };
//		float3 h = gridDiameter / make_float3
//		(
//			solverOptions.gridSize[0],
//			solverOptions.gridSize[1],
//			solverOptions.gridSize[2]
//		);
//
//		float4	dVx = { 0,0,0,0 };
//		float4	dVy = { 0, 0, 0,0 };
//		float4  dVz = { 0, 0, 0 ,0 };
//
//
//
//		for (int i = 0; i < solverOptions.gridSize[1]; i++)
//		{
//			for (int j = 0; j < solverOptions.gridSize[2]; j++)
//			{
//				relativePos = make_float3((float)index, (float)i, (float)j);
//				relativePos = relativePos / gridSize;
//
//
//				dVx = tex3D<float4>(t_VelocityField, relativePos.x + h.x / 2.0, relativePos.y, relativePos.z);
//				dVx -= tex3D<float4>(t_VelocityField, relativePos.x - h.x / 2.0, relativePos.y, relativePos.z);
//
//				dVy = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y + h.x / 2.0, relativePos.z);
//				dVy -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y - h.x / 2.0, relativePos.z);
//
//				dVz = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z + h.x / 2.0);
//				dVz -= tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z - h.x / 2.0);
//
//				// This would give us the Jacobian Matrix
//				dVx = dVx / h.x;
//				dVy = dVy / h.y;
//				dVz = dVz / h.z;
//
//				// Calculate the curl (vorticity vector)
//				float3 vorticity_vec = make_float3
//				(
//					dVy.z - dVz.y,
//					dVz.x - dVx.z,
//					dVx.y - dVy.x
//				);
//				// calculate the vorticity magnitude
//				float vorticity_mag = sqrtf(dot(vorticity_vec, vorticity_vec));
//
//				float4 value = { vorticity_mag,0,0,0 };
//				// Now write it back to the CUDA Surface
//
//				surf3Dwrite(value, s_measure, sizeof(float4) * index, i, j);
//
//			}
//		}
//	}
//	
//}



//__device__ void	Euler_2D
//(
//	const int2& initialGridPosition,
//	float2& finalGridPosition,
//	const int2& gridSize,
//	const float2& gridDiameter,
//	const float& dt,
//	cudaTextureObject_t t_VelocityField_0
//)
//{
//	// find the initial position based on the gridSize and gridDiameter
//	float2 initial_pos = make_float2((float)initialGridPosition.x / (float)gridSize.x, (float)initialGridPosition.y / (float)gridSize.y);
//	initial_pos = initial_pos * gridDiameter;
//
//
//	float4 velocity4D = tex2D<float4>(t_VelocityField_0, initialGridPosition.x, initialGridPosition.y);
//	float2 velocity2D = make_float2(velocity4D.y, velocity4D.z);
//
//	float2 final_pos = (velocity2D * dt) + initial_pos;
//
//
//	// Periodic Boundary Condition
//	if (final_pos.y < 0 )
//	{
//		final_pos.y += gridDiameter.y;
//	}
//	else if(final_pos.y > gridDiameter.y)
//	{
//		final_pos.y -= gridDiameter.y;
//	}
//	
//	// Return the position on the grid
//	finalGridPosition.x = final_pos.x;
//	finalGridPosition.y = final_pos.y;
//
//
//}


__device__ void	RK4Path
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity,
	float3 velocityScale
)
{
	if (particle->outOfScope == false || periodicity)
	{
		//####################### K1 ######################
		float3 k1 = { 0,0,0 };

		float3 relativePos = world2Tex(particle->m_position, gridDiameter, gridSize);

		float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
		float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
		k1 = velocity * dt * velocityScale;


		//####################### K2 ######################
		float3 k2 = { 0,0,0 };

		relativePos = world2Tex(particle->m_position + (k1 * 0.5f), gridDiameter, gridSize);

		velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		k2 = velocity;


		velocity4D = cubicTex3DSimple(t_VelocityField_1, relativePos);
		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		// Using the linear interpolation
		k2 += velocity;
		k2 = k2 / 2.0;
		k2 = dt * k2* velocityScale;

		//####################### K3 ######################
		float3 k3 = { 0,0,0 };

		relativePos = world2Tex(particle->m_position + (k2 * 0.5f), gridDiameter, gridSize);

		velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		k3 = velocity;

		velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		// Using the linear interpolation
		k3 += velocity;
		k3 = k3 / 2.0;
		k3 = dt * k3* velocityScale;

		//####################### K4 ######################

		float3 k4 = { 0,0,0 };

		relativePos = world2Tex(particle->m_position + k3 , gridDiameter, gridSize);
		velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		k4 = dt * velocity* velocityScale;


		float3 newPosition = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);


		if (periodicity)
		{
			particle->m_position = newPosition;
			particle->updateVelocity(gridDiameter, gridSize,t_VelocityField_1);
		}
		else
		{
			if (newPosition < gridDiameter && newPosition > make_float3(0.0f, 0.0f, 0.0f))
			{
				particle->m_position = newPosition;
				particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_1);
			}
			else
			{
				particle->outOfScope = true;
			}
		}

	}

}



__device__ void	PathEuler
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity,
	float3 velocityScale
)
{
	if (particle->outOfScope == false || periodicity)
	{

		float3 newPosition = particle->m_position + dt*particle->m_velocity* velocityScale;;



		if (periodicity)
		{
			particle->m_position = newPosition;
			particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_1);
		}
		else
		{
			if (newPosition < gridDiameter && newPosition > make_float3(0.0f, 0.0f, 0.0f))
			{
				particle->m_position = newPosition;
				particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_1);
			}
			else
			{
				particle->outOfScope = true;
			}
		}

	}

}



__device__ Particle RK4Streak
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	float3 position,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity,
	float3 velocityScale
)
{
	


	//####################### K1 ######################
	float3 k1 = { 0,0,0 };

	float3 relativePos = world2Tex(position, gridDiameter, gridSize);

	float4 velocity4D = cubicTex3DSimple(t_VelocityField_0, relativePos);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
	k1 = velocity * dt * velocityScale;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = world2Tex(position + (k1 * 0.5f), gridDiameter, gridSize);

	velocity4D = cubicTex3DSimple(t_VelocityField_0, relativePos);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k2 = velocity;


	velocity4D = cubicTex3DSimple(t_VelocityField_1, relativePos);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k2 += velocity;
	k2 = k2 / 2.0;
	k2 = dt * k2* velocityScale;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = world2Tex(position + (k2 * 0.5f), gridDiameter, gridSize);

	velocity4D = cubicTex3DSimple(t_VelocityField_0, relativePos);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k3 = velocity;

	velocity4D = cubicTex3DSimple(t_VelocityField_1, relativePos);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	// Using the linear interpolation
	k3 += velocity;
	k3 = k3 / 2.0;
	k3 = dt * k3* velocityScale;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = world2Tex(position + k3, gridDiameter, gridSize);
	velocity4D = cubicTex3DSimple(t_VelocityField_1, relativePos);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity* velocityScale;

	Particle tempParticle;
	tempParticle.m_position = position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);
	tempParticle.updateVelocity(gridDiameter, gridSize, t_VelocityField_1);

	return tempParticle;
}


__device__ void	RK4Path_linear
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	bool periodicity
)
{
	if (particle->outOfScope == false || periodicity)
	{


		//####################### K1 ######################
		float3 k1 = { 0,0,0 };

		float3 relativePos = world2Tex(particle->m_position, gridDiameter, gridSize);

		float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);

		float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };
		k1 = velocity * dt;


		//####################### K2 ######################
		float3 k2 = { 0,0,0 };

		relativePos = world2Tex(particle->m_position + (k1 * 0.5f), gridDiameter, gridSize);

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

		relativePos = world2Tex(particle->m_position + (k2 * 0.5f), gridDiameter, gridSize);

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

		relativePos = world2Tex(particle->m_position + k3, gridDiameter, gridSize);
		velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);

		velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

		k4 = dt * velocity;


		float3 newPosition = particle->m_position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);


		if (periodicity)
		{
			particle->m_position = newPosition;
			particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_1);
		}
		else
		{
			if (newPosition < gridDiameter && newPosition > make_float3(0.0f, 0.0f, 0.0f))
			{
				particle->m_position = newPosition;
				particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_1);
			}
			else
			{
				particle->outOfScope = true;
			}
		}

	}

}


__device__ float3 binarySearch_X
(
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
)
{
	float3 position = _position;
	float3 relative_position = position / gridDiameter;
	float3 samplingStep = _samplingStep * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;

	while (fabsf(ValueAtXYZ_Texture_float4(field, relative_position).x - value) > tolerance&& counter < maxIteration)
	{

		if (ValueAtXYZ_Texture_float4(field, relative_position).x - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
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

			position = position + samplingStep;
			relative_position = position / gridDiameter;
			side = 1;

		}
		counter++;

	}

	return position;

};