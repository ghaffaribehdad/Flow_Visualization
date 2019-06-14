#pragma once
#include "CudaHelper.cuh"
#include "helper_math.cuh"


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