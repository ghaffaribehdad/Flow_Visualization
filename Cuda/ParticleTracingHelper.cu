#include "ParticleTracingHelper.h"
#include "helper_math.h"


__global__ void ParticleTracing::TracingStream
(
	Particle* d_particles,
	cudaTextureObject_t t_VelocityField,
	SolverOptions solverOptions,
	Vertex* p_VertexBuffer
)
{

	unsigned int particleIdx = CUDA_INDEX;

	if (particleIdx < solverOptions.lines_count)
	{

		int vertexIdx = particleIdx * solverOptions.lineLength;

		float dt = solverOptions.dt;

		float3 gridDiameter = Array2Float3(solverOptions.gridDiameter);
		int3 gridSize = Array2Int3(solverOptions.gridSize);
		float3 init_pos = *d_particles[particleIdx].getPosition();
		float3 upDir = make_float3(0.0f, 1.0f, 0.0f);
		d_particles[particleIdx].updateVelocity(gridDiameter, gridSize, t_VelocityField, Array2Float3(solverOptions.velocityScalingFactor));

		for (int i = 0; i < solverOptions.lineLength; i++)
		{


			p_VertexBuffer[vertexIdx + i].initialPos.x = init_pos.x;
			p_VertexBuffer[vertexIdx + i].initialPos.y = init_pos.y;
			p_VertexBuffer[vertexIdx + i].initialPos.z = init_pos.z;


			// Center the position of the particles
			p_VertexBuffer[vertexIdx + i].pos.x = d_particles[particleIdx].getPosition()->x;
			p_VertexBuffer[vertexIdx + i].pos.y = d_particles[particleIdx].getPosition()->y;
			p_VertexBuffer[vertexIdx + i].pos.z = d_particles[particleIdx].getPosition()->z;

			float3 current_vel = d_particles[particleIdx].m_velocity;

			float3 tangent = normalize(current_vel);

			p_VertexBuffer[vertexIdx + i].normal.x = upDir.x;
			p_VertexBuffer[vertexIdx + i].normal.y = upDir.y;
			p_VertexBuffer[vertexIdx + i].normal.z = upDir.z;

			p_VertexBuffer[vertexIdx + i].tangent.x = tangent.x;
			p_VertexBuffer[vertexIdx + i].tangent.y = tangent.y;
			p_VertexBuffer[vertexIdx + i].tangent.z = tangent.z;

			p_VertexBuffer[vertexIdx + i].LineID = particleIdx;
			p_VertexBuffer[vertexIdx + i].time = i;
			//updates velocity and position of the particle 
			ParticleTracing::RK4Stream(t_VelocityField, &d_particles[particleIdx], gridDiameter, Array2Int3(solverOptions.gridSize), dt, Array2Float3(solverOptions.velocityScalingFactor));

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
				float3 gamma2 = (d_particles[particleIdx].m_velocity - gamma1) / dt;
				float3 gamma1Xgamma2 = cross(gamma1, gamma2);
				float curvature = magnitude(gamma1Xgamma2);
				curvature = curvature / powf(magnitude(gamma1), 3);
				p_VertexBuffer[vertexIdx + i].measure = curvature;

				break;
			}

			case ColorMode::ColorMode::DISTANCE_STREAK:
			{

				p_VertexBuffer[vertexIdx + i].measure = fabs(magnitude(init_pos - d_particles[particleIdx].m_position));

				break;
			}

			}
		}
	}
}


__device__ float3 ParticleTracing::RK4
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
	float3 relativePos = world2Tex(position, gridDiameter, gridSize, false, true);
	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z };


	// (h x k1)
	float3 k1 = velocity * dt * velocityScale;


	//####################### K2 ######################
	relativePos = world2Tex(position + (k1 * 0.5f), gridDiameter, gridSize, false, true);
	velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);

	float3 k21 = { velocity4D.x,velocity4D.y,velocity4D.z };

	velocity4D = tex3D<float4>(t_VelocityField_1, relativePos.x, relativePos.y, relativePos.z);
	float3 k22 = { velocity4D.x, velocity4D.y, velocity4D.z };

	// Average of K11 and K12
	float3 k2 = 0.5 * (k21 + k22);

	// (h x k2)
	k2 = 0.5 * dt * k2 * velocityScale;
	//####################### K3 ######################
	relativePos = world2Tex(position + (k2* 0.5f), gridDiameter, gridSize, false, true);

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

__device__ void ParticleTracing::RK4Stream(
	cudaTextureObject_t t_VelocityField_0,
	Particle* particle,
	const float3& gridDiameter,
	const int3& gridSize,
	float dt,
	float3 velocityScale
)
{

	particle->m_position = ParticleTracing::RK4(t_VelocityField_0, t_VelocityField_0, particle->m_position, gridDiameter, gridSize, dt, velocityScale);

	float3 relativePos = world2Tex(particle->m_position, gridDiameter, gridSize, false, true);

	float4 velocity4D = tex3D<float4>(t_VelocityField_0, relativePos.x, relativePos.y, relativePos.z);
	particle->m_velocity = make_float3(velocity4D.x, velocity4D.y, velocity4D.z) *velocityScale;
}