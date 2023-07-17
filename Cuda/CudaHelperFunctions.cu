#include "CudaHelperFunctions.h"
#include <random>
#include "helper_math.h"
#include "../Options/SolverOptions.h"
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
	float3 relativePos = world2Tex(position, gridDiameter, gridSize, false,true);
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };


	// (h x k1)
	float3 k1 = velocity * dt * velocityScale;


	//####################### K2 ######################
	relativePos = world2Tex(position + (k1 * 0.5f), gridDiameter, gridSize, false, true);
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);

	float3 k21 = { velocity4D.x,velocity4D.y,velocity4D.z };

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	float3 k22 = {velocity4D.x, velocity4D.y, velocity4D.z };

	// Average of K11 and K12
	float3 k2 = 0.5 * (k21 + k22);

	// (h x k2)
	k2 = 0.5 * dt * k2 * velocityScale;
	//####################### K3 ######################
	relativePos = world2Tex(position +  (k2* 0.5f), gridDiameter, gridSize, false, true);

	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 k31 = { velocity4D.x,velocity4D.y,velocity4D.z };

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	float3 k32 = { velocity4D.x, velocity4D.y, velocity4D.z };

	float3 k3 = 0.5 * (k31 + k32);

	// (h x k3)
	k3 = dt * k3 * velocityScale; 

	//####################### K4 ######################
	relativePos = world2Tex(position + k3, gridDiameter, gridSize, false, true);

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	velocity = { velocity4D.x,velocity4D.y,velocity4D.z };

	float3 k4 = dt * velocity * velocityScale;


	// y0 + 1/6 dt (k1 + 2 k2 + 2 k3 + k4)	
	return position + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}


__global__ void TracingStreak(cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int particleIdx = CUDA_INDEX;


	if (particleIdx < solverOptions.lines_count)
	{
		// line_index indicates the line segment index


		float timeSteps = solverOptions.lastIdx - solverOptions.firstIdx + 1;



		int lineIndex = particleIdx * solverOptions.lineLength;
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);





		for (int i = 0; i <= step; i++)
		{
			float3 oldPos = { 0,0,0 };
			Particle tempPar;

			oldPos = make_float3
			(
				p_VertexBuffer[lineIndex + i].pos.x ,
				p_VertexBuffer[lineIndex + i].pos.y ,
				p_VertexBuffer[lineIndex + i].pos.z
			);
	

			if (odd)
			{
				tempPar = RK4Streak(t_VelocityField_1, t_VelocityField_0, oldPos, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

			}
			else //Even
			{

				tempPar = RK4Streak(t_VelocityField_0, t_VelocityField_1, oldPos, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

			}

			// Update Positions
			p_VertexBuffer[lineIndex + i].pos.x = tempPar.m_position.x ;
			p_VertexBuffer[lineIndex + i].pos.y = tempPar.m_position.y ;
			p_VertexBuffer[lineIndex + i].pos.z = tempPar.m_position.z ;

			p_VertexBuffer[lineIndex + i].initialPos.x = p_VertexBuffer[lineIndex].pos.x;
			p_VertexBuffer[lineIndex + i].initialPos.y = p_VertexBuffer[lineIndex].pos.y;
			p_VertexBuffer[lineIndex + i].initialPos.z = p_VertexBuffer[lineIndex].pos.z;

			if (!solverOptions.projectToInit)
			{
				switch (solverOptions.projection)
				{
				case Projection::ZY_PROJECTION:
				{
					
					if (solverOptions.syncWithStreak)
					{
						float init_pos = (gridDiameter / gridSize).x * solverOptions.projectPos;
						init_pos -= solverOptions.timeDim / 2;
						init_pos += (solverOptions.currentIdx - solverOptions.firstIdx) * (solverOptions.timeDim / timeSteps);
						p_VertexBuffer[lineIndex + i].initialPos.x = init_pos;

					}
					break;
				}
				default:
					break;
				}
			}



			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
			float3 tangent = normalize(tempPar.m_velocity);


			

			p_VertexBuffer[lineIndex + i].normal.x = upDir.x;
			p_VertexBuffer[lineIndex + i].normal.y = upDir.y;
			p_VertexBuffer[lineIndex + i].normal.z = upDir.z;


			p_VertexBuffer[lineIndex + i].tangent.x = tangent.x;
			p_VertexBuffer[lineIndex + i].tangent.y = tangent.y;
			p_VertexBuffer[lineIndex + i].tangent.z = tangent.z;


			////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/////////////
			p_VertexBuffer[lineIndex + i].LineID = particleIdx;

			switch (solverOptions.colorMode)
			{
			case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
			{
				p_VertexBuffer[lineIndex + i].measure = magnitude(tempPar.m_velocity);
				break;

			}
			case ColorMode::ColorMode::U_VELOCITY: // Vx
			{

				p_VertexBuffer[lineIndex + i].measure = tempPar.getVelocity()->x;
				break;
			}
			case ColorMode::ColorMode::V_VELOCITY: // Vy
			{
				p_VertexBuffer[lineIndex + i].measure = tempPar.getVelocity()->y;
				break;
			}
			case ColorMode::ColorMode::W_VELOCITY: // Vz
			{
				p_VertexBuffer[lineIndex + i].measure = tempPar.getVelocity()->z;
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


//__global__ void applyGaussianFilter
//(
//	int filterSize,
//	int3 fieldSize,
//	cudaTextureObject_t t_velocityField,
//	cudaSurfaceObject_t	s_velocityField
//)
//{
//	int index = CUDA_INDEX;
//
//	if (index < fieldSize.y)
//	{
//
//
//			float * filter = gaussianFilter2D(filterSize, 1);
//			applyFilter2D(filter, filterSize, t_velocityField, s_velocityField, 2, index, fieldSize);
//			delete[] filter;
//
//	}
//}



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

__device__ void writeBufferPathline(Vertex * vertex, SolverOptions * solverOptions, Particle & particle, float3 & tangent, float3 & up,float3 currentVelocity = { 0,0,0 })
{
	vertex->normal.x = up.x;
	vertex->normal.y = up.y;
	vertex->normal.z = up.z;


	vertex->tangent.x = tangent.x;
	vertex->tangent.y = tangent.y;
	vertex->tangent.z = tangent.z;

	switch (solverOptions->colorMode)
	{
	case ColorMode::ColorMode::VELOCITY_MAG: // Velocity
	{
		vertex->measure = magnitude(particle.m_velocity);
		break;

	}
	case ColorMode::ColorMode::U_VELOCITY: // Vx
	{

		vertex->measure = particle.getVelocity()->x;
		break;
	}
	case ColorMode::ColorMode::V_VELOCITY: // Vy
	{
		vertex->measure = particle.getVelocity()->y;
		break;
	}
	case ColorMode::ColorMode::W_VELOCITY: // Vz
	{
		vertex->measure = particle.getVelocity()->z;
		break;

	}
	case ColorMode::ColorMode::CURVATURE:
	{
		float3 gamma1 = currentVelocity;
		float3 gamma2 = (particle.m_velocity - gamma1) / Array2Float3(solverOptions->velocityScalingFactor);
		float3 gamma1Xgamma2 = cross(gamma1, gamma2);
		float curvature = magnitude(gamma1Xgamma2);
		curvature = curvature / powf(magnitude(gamma1), 3);
		vertex->measure = curvature;

	}
	}
}



__global__ void TracingPath(Particle* d_particles, cudaTextureObject_t t_VelocityField_0, cudaTextureObject_t t_VelocityField_1, SolverOptions solverOptions, Vertex* p_VertexBuffer, bool odd, int step)
{
	int particleIdx = CUDA_INDEX;
	if (particleIdx < solverOptions.lines_count)
	{	
		float timeSteps = solverOptions.lastIdx - solverOptions.firstIdx + 1;
		// line_index indicates the line segment index
		int lineIdx = particleIdx * solverOptions.lineLength;
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);
		d_particles[particleIdx].updateVelocity(gridDiameter, gridSize, t_VelocityField_0);
		float3 current_vel = d_particles[particleIdx].m_velocity;
		// Initialization
		if (step == 0)
		{


			for (int i = 0; i < solverOptions.lineLength; i++)
			{

				p_VertexBuffer[lineIdx + i].initialPos.x = d_particles[particleIdx].getPosition()->x;
				p_VertexBuffer[lineIdx + i].initialPos.y = d_particles[particleIdx].getPosition()->y;
				p_VertexBuffer[lineIdx + i].initialPos.z = d_particles[particleIdx].getPosition()->z;
			}
			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
			float3 tangent = normalize(d_particles[particleIdx].m_velocity);
			p_VertexBuffer[lineIdx + step].LineID = particleIdx;
			p_VertexBuffer[lineIdx + step].pos.x = d_particles[particleIdx].getPosition()->x;
			p_VertexBuffer[lineIdx + step].pos.y = d_particles[particleIdx].getPosition()->y;
			p_VertexBuffer[lineIdx + step].pos.z = d_particles[particleIdx].getPosition()->z;
			writeBufferPathline(&p_VertexBuffer[lineIdx + step], &solverOptions, d_particles[particleIdx], tangent, upDir, current_vel);
		}

		else
		{
			if (!solverOptions.projectToInit)
			{
				switch (solverOptions.projection)
				{
				case Projection::ZY_PROJECTION:
				{
					if (solverOptions.syncWithStreak)
					{

						float init_pos = (gridDiameter / gridSize).x * solverOptions.projectPos;
						init_pos -= solverOptions.timeDim / 2;
						init_pos += (solverOptions.currentIdx - solverOptions.firstIdx) * (solverOptions.timeDim / timeSteps);

						for (int i = 0; i < solverOptions.lineLength; i++)
						{
							p_VertexBuffer[lineIdx + i].initialPos.x = init_pos;
						}

					}
					break;
				}

				}
			}

			switch (odd)
			{
			case true: // Odd integration steps (1,3,5,...)
			{
				RK4Path(t_VelocityField_0, t_VelocityField_1, &d_particles[particleIdx], gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

				break;
			}
			case false:// Even integration steps (2,4,6,...)
			{
				RK4Path(t_VelocityField_1, t_VelocityField_0, &d_particles[particleIdx], gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

				break;
			}

			}
			// use the up vector as normal
			float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
			float3 tangent = normalize(d_particles[particleIdx].m_velocity);

			// Write into the Vertex BUffer
			p_VertexBuffer[lineIdx + step].pos.x = d_particles[particleIdx].getPosition()->x;
			p_VertexBuffer[lineIdx + step].pos.y = d_particles[particleIdx].getPosition()->y;
			p_VertexBuffer[lineIdx + step].pos.z = d_particles[particleIdx].getPosition()->z;
			p_VertexBuffer[lineIdx + step].LineID = particleIdx;
			p_VertexBuffer[lineIdx + step].time = step;

			writeBufferPathline(&p_VertexBuffer[lineIdx + step], &solverOptions, d_particles[particleIdx], tangent, upDir, current_vel);
			
		}
	}
}




__global__ void TracingPathSurface(
	cudaTextureObject_t t_VelocityField_0,
	cudaTextureObject_t t_VelocityField_1,
	cudaSurfaceObject_t s_PathSpaceTime,
	SolverOptions solverOptions,
	PathSpaceTimeOptions pathSpaceTimeOptions,
	int step){

	int index = CUDA_INDEX;

	int3 particleIndex = indexto3D(Array2Int3(pathSpaceTimeOptions.seedGrid), index);
	float3 position3 = make_float3(0, 0, 0);
	int time_length = pathSpaceTimeOptions.lastIdx - pathSpaceTimeOptions.firstIdx;
	int time_batch = time_length / pathSpaceTimeOptions.timeGrid;
	int t_step = int(step / time_batch);

	if (index < pathSpaceTimeOptions.seedGrid[0] * pathSpaceTimeOptions.seedGrid[1] * pathSpaceTimeOptions.seedGrid[2]) {

		// line_index indicates the line segment index
		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);

		// Fetch initial position
		float4 position4 = make_float4(0, 0, 0, 0);

		if (step == 0) {
			surf3Dread(&position4, s_PathSpaceTime, particleIndex.x * sizeof(float4), particleIndex.y, particleIndex.z + (t_step)* pathSpaceTimeOptions.seedGrid[2]);
		} else  if (step % time_batch == 0) {
			surf3Dread(&position4, s_PathSpaceTime, particleIndex.x * sizeof(float4), particleIndex.y, particleIndex.z + (t_step - 1)* pathSpaceTimeOptions.seedGrid[2]);
		} else {
			surf3Dread(&position4, s_PathSpaceTime, particleIndex.x * sizeof(float4), particleIndex.y, particleIndex.z + (t_step)* pathSpaceTimeOptions.seedGrid[2]);
		}

		position3 = make_float3(position4.x,position4.y,position4.z);
		// trace particle in field
		if(step % 2 == 0)
			position3 = RK4(t_VelocityField_0, t_VelocityField_1, position3, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));
		else
			position3 = RK4(t_VelocityField_1, t_VelocityField_0, position3, gridDiameter, gridSize, solverOptions.dt, Array2Float3(solverOptions.velocityScalingFactor));

		// write back the new position
		position4 =make_float4(position3.x, position3.y, position3.z,0);
		surf3Dwrite(position4, s_PathSpaceTime, sizeof(float4) * particleIndex.x, particleIndex.y, particleIndex.z + t_step * pathSpaceTimeOptions.seedGrid[2]);

	}
}




__global__ void initializePathSpaceTime(Particle * d_particle, cudaSurfaceObject_t s_pathSpaceTime, PathSpaceTimeOptions pathSpaceTimeOptions){
	
	int index = CUDA_INDEX;
	int3 particleIndex = indexto3D(Array2Int3(pathSpaceTimeOptions.seedGrid), index);
	float4 position4 = { 0,0,0,0 };
	float3 position3 = { 0,0,0 };
	int3 seedGrid = Array2Int3(pathSpaceTimeOptions.seedGrid);

	if (index < seedGrid.x * seedGrid.y* seedGrid.z){

		int linear_index = particleIndex.x * seedGrid.y * seedGrid.z + particleIndex.y * seedGrid.z + particleIndex.z;
		position3 = *d_particle[linear_index].getPosition();
		position4 = make_float4(position3.x, position3.y, position3.z, 0);
		surf3Dwrite(position4, s_pathSpaceTime, sizeof(float4) * particleIndex.x, particleIndex.y, particleIndex.z);
		//printf("particle %d is at (%3f,%3f,%3f) and index are:  (%d,%d,%d) \n", index, position4.x, position4.y, position4.z, particleIndex.x, particleIndex.y, particleIndex.z);

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


