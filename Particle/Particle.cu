#include "Particle.h"
#include "..//Cuda/helper_math.h"


__device__  void Particle::updateVelocity(const float3& gridDiameter, const int3& gridSize, cudaTextureObject_t t_VelocityField, float3 velocityScale )
{
	float3 relativePos = world2Tex(m_position, gridDiameter, gridSize, false,true);
	float4 velocity4D = tex3D<float4>(t_VelocityField, relativePos.x, relativePos.y, relativePos.z);
	float3 velocity = { velocity4D.x,velocity4D.y,velocity4D.z } ;
	velocity = velocity * velocityScale;
	this->setVelocity(velocity);

}



