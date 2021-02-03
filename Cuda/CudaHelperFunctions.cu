#include "CudaHelperFunctions.h"
#include <random>
#include "helper_math.h"

__device__ float3 RK4
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	const float3 & position,
	const float3 & gridDiameter,
	const int3 & gridSize,
	const float & dt,
	const float3 & velocityScale
)
{
	//####################### K1 ######################
	float3 k1 = { 0,0,0 };
	float3 relativePos = world2Tex(position, gridDiameter, gridSize, true);
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);

	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k1 = velocity * dt * velocityScale;


	//####################### K2 ######################
	float3 k2 = { 0,0,0 };

	relativePos = world2Tex(position + (k1 * 0.5f), gridDiameter, gridSize, true);
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	k2 = { velocity4D.x,velocity4D.y,velocity4D.z };


	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	k2 = { k2.x +velocity4D.x,k2.y + velocity4D.y,k2.z + velocity4D.z };
	k2 = k2 * 0.5;


	k2 = dt * k2 * velocityScale;

	//####################### K3 ######################
	float3 k3 = { 0,0,0 };

	relativePos = world2Tex(position + (k2* 0.5f), gridDiameter, gridSize, true);

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	k3 = { velocity4D.x,velocity4D.y,velocity4D.z };

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	k3 = { k3.x + velocity4D.x,k3.y + velocity4D.y,k3.z + velocity4D.z };
	k3 = k2 * 0.5;


	k3 = dt * k3 * velocityScale;

	//####################### K4 ######################

	float3 k4 = { 0,0,0 };

	relativePos = world2Tex(position + k3, gridDiameter, gridSize, true);

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	k4 = dt * velocity * velocityScale;

	return position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2 * k3 + k4);
}

__device__ void RK4Stream(
	cudaTextureObject_t t_VelocityField_0,
	Particle* particle,
	const float3& gridDiameter,
	const int3& gridSize,
	float dt,
	float3 velocityScale
)
{

	particle->m_position = RK4(t_VelocityField_0, t_VelocityField_0, particle->m_position, gridDiameter, gridSize, dt, velocityScale);
	particle->updateVelocity(gridDiameter, gridSize, t_VelocityField_0);
	
}







__global__ void TracingStreak(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int particleIdx = blockDim.x * blockIdx.x + threadIdx.x;


	if (particleIdx < solverOptions.lines_count)
	{
		// line_index indicates the line segment index
		int vertexIdx = particleIdx * solverOptions.lineLength;
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);





		for (int i = 0; i <= step; i++)
		{
			float3 oldPos = { 0,0,0 };
			Particle tempPar;

			oldPos = make_float3
			(
				p_VertexBuffer[vertexIdx + i].pos.x ,
				p_VertexBuffer[vertexIdx + i].pos.y ,
				p_VertexBuffer[vertexIdx + i].pos.z
			);
	

			if (odd)
			{
				tempPar = RK4Streak(t_VelocityField_1, t_VelocityField_0, oldPos, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

			}
			else //Even
			{

				tempPar = RK4Streak(t_VelocityField_0, t_VelocityField_1, oldPos, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

			}

			p_VertexBuffer[vertexIdx + i].pos.x = tempPar.m_position.x ;
			p_VertexBuffer[vertexIdx + i].pos.y = tempPar.m_position.y ;
			p_VertexBuffer[vertexIdx + i].pos.z = tempPar.m_position.z ;


			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
			float3 tangent = normalize(tempPar.m_velocity);


			

			p_VertexBuffer[vertexIdx + i].normal.x = upDir.x;
			p_VertexBuffer[vertexIdx + i].normal.y = upDir.y;
			p_VertexBuffer[vertexIdx + i].normal.z = upDir.z;


			p_VertexBuffer[vertexIdx + i].tangent.x = tangent.x;
			p_VertexBuffer[vertexIdx + i].tangent.y = tangent.y;
			p_VertexBuffer[vertexIdx + i].tangent.z = tangent.z;


			////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/////////////
			p_VertexBuffer[vertexIdx + i].LineID = particleIdx;

			switch (solverOptions.colorMode)
			{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{
				p_VertexBuffer[vertexIdx + i].measure = magnitude(tempPar.m_velocity);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY: // Vx
			{

				p_VertexBuffer[vertexIdx + i].measure = tempPar.getVelocity()->x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY: // Vy
			{
				p_VertexBuffer[vertexIdx + i].measure = tempPar.getVelocity()->y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY: // Vz
			{
				p_VertexBuffer[vertexIdx + i].measure = tempPar.getVelocity()->z;
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
			p_VertexBuffer[line_index + i].pos.x = d_particles[index].getPosition()->x ;
			p_VertexBuffer[line_index + i].pos.y = d_particles[index].getPosition()->y ;
			p_VertexBuffer[line_index + i].pos.z = d_particles[index].getPosition()->z ;
			p_VertexBuffer[line_index + i].initialPos.x = d_particles[index].getPosition()->x;
			p_VertexBuffer[line_index + i].initialPos.y = d_particles[index].getPosition()->y;
			p_VertexBuffer[line_index + i].initialPos.z = d_particles[index].getPosition()->z;
		}

	}
}


__global__ void applyGaussianFilter
(
	int filterSize,
	int3 fieldSize,
	cudaTextureObject_t t_velocityField,
	cudaSurfaceObject_t	s_velocityField
)
{
	int index = CUDA_INDEX;

	if (index < fieldSize.y)
	{


			float * filter = gaussianFilter2D(filterSize, 1);
			applyFilter2D(filter, filterSize, t_velocityField, s_velocityField, 2, index, fieldSize);
			delete[] filter;

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
			p_VertexBuffer[line_index + i].pos.x = p_VertexBuffer[line_index + i].pos.x;
			p_VertexBuffer[line_index + i].pos.y = p_VertexBuffer[line_index + i].pos.y ;
			p_VertexBuffer[line_index + i].pos.z = p_VertexBuffer[line_index + i].pos.z ;
		}

	}

}

__global__ void TracingPath(Particle* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int particleIdx = blockDim.x * blockIdx.x + threadIdx.x;


	if (particleIdx < solverOptions.lines_count)
	{
		
		// line_index indicates the line segment index
		int lineIdx = particleIdx * solverOptions.lineLength;

		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);

		float3 current_vel = d_particles[particleIdx].m_velocity;
		
		if (step == 0)
		{
			for (int i = 0; i < solverOptions.lineLength; i++)
			{
				p_VertexBuffer[lineIdx + i].initialPos.x = d_particles[particleIdx].getPosition()->x ;
				p_VertexBuffer[lineIdx + i].initialPos.y = d_particles[particleIdx].getPosition()->y ;
				p_VertexBuffer[lineIdx + i].initialPos.z = d_particles[particleIdx].getPosition()->z ;
			}
		}

		if (odd)
		{
			RK4Path(t_VelocityField_1, t_VelocityField_0, &d_particles[particleIdx], gridDiameter, gridSize, solverOptions.dt,Array2Float3(solverOptions.velocityScalingFactor));
		}
		else //Even
		{

			RK4Path(t_VelocityField_0, t_VelocityField_1, &d_particles[particleIdx], gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));
		}


		// use the up vector as normal
		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);


		float3 tangent = normalize(d_particles[particleIdx].m_velocity);




		// Write into the Vertex BUffer
		p_VertexBuffer[lineIdx + step].pos.x = d_particles[particleIdx].getPosition()->x ;
		p_VertexBuffer[lineIdx + step].pos.y = d_particles[particleIdx].getPosition()->y ;
		p_VertexBuffer[lineIdx + step].pos.z = d_particles[particleIdx].getPosition()->z ;

		p_VertexBuffer[lineIdx + step].normal.x = upDir.x;
		p_VertexBuffer[lineIdx + step].normal.y = upDir.y;
		p_VertexBuffer[lineIdx + step].normal.z = upDir.z;


		p_VertexBuffer[lineIdx + step].tangent.x = tangent.x;
		p_VertexBuffer[lineIdx + step].tangent.y = tangent.y;
		p_VertexBuffer[lineIdx + step].tangent.z = tangent.z;

		p_VertexBuffer[lineIdx + step].LineID = particleIdx;

		switch (solverOptions.colorMode)
		{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{
				p_VertexBuffer[lineIdx + step].measure = magnitude(d_particles[particleIdx].m_velocity);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY: // Vx
			{

				p_VertexBuffer[lineIdx + step].measure = d_particles[particleIdx].getVelocity()->x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY: // Vy
			{
				p_VertexBuffer[lineIdx + step].measure = d_particles[particleIdx].getVelocity()->y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY: // Vz
			{
				p_VertexBuffer[lineIdx + step].measure = d_particles[particleIdx].getVelocity()->z;
				break;

			}
			case ColorMode::ColorMode::CURVATURE:
			{
				float3 gamma1 = current_vel;
				float3 gamma2 = (d_particles[particleIdx].m_velocity - gamma1) / Array2Float3(solverOptions.velocityScalingFactor);
				float3 gamma1Xgamma2 = cross(gamma1, gamma2);
				float curvature = magnitude(gamma1Xgamma2);
				curvature = curvature / powf(magnitude(gamma1), 3);
				p_VertexBuffer[lineIdx + step].measure = curvature;

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
	unsigned int particleIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (particleIdx < solverOptions.lines_count)
	{
		int vertexIdx = particleIdx * solverOptions.lineLength;
		float dt = solverOptions.dt;


		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridsize = Array2Int3(solverOptions.gridSize);


		// Update the velocity of the particle at its initial position
		d_particles[particleIdx].updateVelocity(gridDiameter, gridsize, t_VelocityField);

		float3 init_pos = *d_particles[particleIdx].getPosition();


		// Up direction is needed for Shading (Must be revised)
		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);



		for (int i = 0; i < solverOptions.lineLength; i++)
		{


			p_VertexBuffer[vertexIdx + i].initialPos.x = init_pos.x;
			p_VertexBuffer[vertexIdx + i].initialPos.y = init_pos.y ;
			p_VertexBuffer[vertexIdx + i].initialPos.z = init_pos.z ;


			// Center the position of the particles
			p_VertexBuffer[vertexIdx + i].pos.x = d_particles[particleIdx].getPosition()->x ;
			p_VertexBuffer[vertexIdx + i].pos.y = d_particles[particleIdx].getPosition()->y ;
			p_VertexBuffer[vertexIdx + i].pos.z = d_particles[particleIdx].getPosition()->z ;

			float3 current_vel = d_particles[particleIdx].m_velocity;

			float3 tangent = normalize(current_vel);

			p_VertexBuffer[vertexIdx + i].normal.x = upDir.x;
			p_VertexBuffer[vertexIdx + i].normal.y = upDir.y;
			p_VertexBuffer[vertexIdx + i].normal.z = upDir.z;

			p_VertexBuffer[vertexIdx + i].tangent.x = tangent.x;
			p_VertexBuffer[vertexIdx + i].tangent.y = tangent.y;
			p_VertexBuffer[vertexIdx + i].tangent.z = tangent.z;

			p_VertexBuffer[vertexIdx + i].LineID = particleIdx;

			//updates velocity and position of the particle 
			RK4Stream(t_VelocityField, &d_particles[particleIdx], gridDiameter, Array2Int3(solverOptions.gridSize), dt, Array2Float3(solverOptions.velocityScalingFactor));

			switch (solverOptions.colorMode)
			{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{

				p_VertexBuffer[vertexIdx + i].measure = magnitude(current_vel);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY:
			{
				p_VertexBuffer[vertexIdx + i].measure = current_vel.x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY:
			{
				p_VertexBuffer[vertexIdx + i].measure = current_vel.y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY:
			{
				p_VertexBuffer[vertexIdx + i].measure = current_vel.z;
				break;
			}
			case ColorMode::ColorMode::CURVATURE:
			{
				float3 gamma1 = current_vel;
				float3 gamma2 = (d_particles[particleIdx].m_velocity - gamma1)/dt;
				float3 gamma1Xgamma2 = cross(gamma1, gamma2);
				float curvature = magnitude(gamma1Xgamma2);
				curvature = curvature / powf(magnitude(gamma1), 3);
				p_VertexBuffer[vertexIdx + i].measure = curvature;

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


__device__ void	RK4Path
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	Particle* particle,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	float3 velocityScale
)
{

	float3 newPosition =  RK4(t_VelocityField_0, t_VelocityField_1, particle->m_position, gridDiameter, gridSize, dt, velocityScale);
	particle->m_position = newPosition;
	particle->updateVelocity(gridDiameter, gridSize,t_VelocityField_1);
	
}


__device__ Particle RK4Streak
(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	float3 position,
	float3 gridDiameter,
	int3 gridSize,
	float dt,
	float3 velocityScale
)
{
	



	Particle tempParticle;
	tempParticle.m_position = RK4(t_VelocityField_0, t_VelocityField_1, position, gridDiameter, gridSize, dt, velocityScale);
	tempParticle.updateVelocity(gridDiameter, gridSize, t_VelocityField_1);



	return tempParticle;
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



__device__ inline float* gaussianFilter2D(int size, float std)
{

	float * filter = new float[size*size];

	// for now x and y must be odd


	for (int i = 0; i < size ; i++)
	{
		for (int j = 0; j < size ; j++)
		{
			filter[i*size + j] = 0.5 * (1 / CUDA_PI_D) * (1.0 / powf(std, 2)) * exp(-1.0 * ((powf(i - size / 2, 2) + powf(j -size / 2, 2)) / (2 * std * std)));
		}
	}


	return filter;
}


__device__ inline void applyFilter2D(float * filter, int filterSize, cudaTextureObject_t tex, cudaSurfaceObject_t surf, int direction, int plane, int3 gridSize)
{




	switch(direction)
	{
	case 0: //XY
	{
		for (int i = 0; i < gridSize.x; i++)
		{
			for (int j = 0; j < gridSize.y; j++)
			{

				float4 filteredValue = { 0,0,0,0 };
				for (int ii = -filterSize / 2; ii <= filterSize / 2; ii++)
				{
					for (int jj = -filterSize / 2; jj <= filterSize / 2; jj++)
					{
						filteredValue = filter[ii*filterSize + jj] * tex3D<float4>(tex, i+ii, j+jj, plane);
					}
				}

				surf3Dwrite(filteredValue, surf, sizeof(float4) * i, j, plane);
				
			}
		}

		break;
	}

	case 1: //YZ
	{
		for (int i = 0; i < gridSize.y; i++)
		{
			for (int j = 0; j < gridSize.z; j++)
			{

				float4 filteredValue = { 0,0,0,0 };

				for (int ii = -filterSize / 2; ii <= filterSize / 2; ii++)
				{
					for (int jj = -filterSize / 2; jj <= filterSize / 2; jj++)
					{
						filteredValue = filter[ii*filterSize + jj] * tex3D<float4>(tex, plane, i + ii, j + jj);

					}
				}

				surf3Dwrite(filteredValue, surf, sizeof(float4) * plane, i, j);



			}
		}

		break;
	}

	case 2: //ZX
	{
		for (int i = 0; i < gridSize.z; i++)
		{
			for (int j = 0; j < gridSize.x; j++)
			{

				float4 filteredValue = { 0,0,0,0 };
				for (int ii = 0; ii < filterSize ; ii++)
				{
					for (int jj = 0 ; jj < filterSize ; jj++)
					{
						filteredValue = filteredValue + filter[ii*filterSize + jj] * tex3D<float4>(tex, j + jj - filterSize/2, plane, i + ii - filterSize / 2);
					}
				}

				surf3Dwrite(filteredValue, surf, sizeof(float4) * j, plane, i);


			}
		}

		break;
	}

	}
}