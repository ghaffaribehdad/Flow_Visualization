
#include "IsosurfaceHelperFunctions.h"
#include "cuda_runtime.h"



__device__ float4 fetchTexture3D(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z);
}

inline __device__  float3 IsosurfaceHelper::Measure::GradientAtXYZ(cudaTextureObject_t tex, float3 position, float h)
{
	float dV_dX = ValueAtXYZ(tex, make_float3(position.x + h / 2.0f, position.y, position.z));
	float dV_dY = ValueAtXYZ(tex, make_float3(position.x, position.y + h / 2.0f, position.z));
	float dV_dZ = ValueAtXYZ(tex, make_float3(position.x, position.y, position.z + h / 2.0f));

	dV_dX -= ValueAtXYZ(tex, make_float3(position.x - h / 2.0f, position.y, position.z));
	dV_dY -= ValueAtXYZ(tex, make_float3(position.x, position.y - h / 2.0f, position.z));
	dV_dZ -= ValueAtXYZ(tex, make_float3(position.x, position.y, position.z - h / 2.0f));

	return { dV_dX / h ,dV_dY / h, dV_dZ / h };
}

__device__ float IsosurfaceHelper::Velocity_Magnitude::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 _velocity = fetchTexture3D(tex,position);
	float3 velocity = make_float3(_velocity.x, _velocity.y, _velocity.z);
	return fabsf(sqrtf(dot(velocity, velocity)));
}

__device__ float IsosurfaceHelper::Velocity_X::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 value =	tex3D<float4>(tex, position.x, position.y, position.z);
	return value.x;
}

__device__ float IsosurfaceHelper::Velocity_Y::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).y;
}

__device__ float IsosurfaceHelper::Velocity_Z::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).z;
}


