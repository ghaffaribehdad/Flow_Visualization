
#include "IsosurfaceHelperFunctions.h"
#include "cuda_runtime.h"
#include "..//Cuda/helper_math.h"


__device__ float3 IsosurfaceHelper::Observable::GradientAtXYZ(cudaTextureObject_t tex, float3 relativePos, int3 gridSize)
{
	int3 gridPoint_0 = floorMult(relativePos, gridSize);
	int3 gridPoint_1 = ceilMult(relativePos, gridSize);


	return { 0,0,0 };
}

__device__  float3 IsosurfaceHelper::Observable::GradientAtGrid(cudaTextureObject_t tex, float3 position, int3 gridSize)
{
	float3 h = { 1.0f, 1.0f ,1.0f };
	h = h/gridSize;
	float dV_dX = this->ValueAtXYZ(tex, make_float3(position.x + h.x / 2.0f, position.y, position.z));
	float dV_dY = this->ValueAtXYZ(tex, make_float3(position.x, position.y + h.y / 2.0f, position.z));
	float dV_dZ = this->ValueAtXYZ(tex, make_float3(position.x, position.y, position.z + h.z / 2.0f));

	dV_dX -= this->ValueAtXYZ(tex, make_float3(position.x - h.x / 2.0f, position.y, position.z));
	dV_dY -= this->ValueAtXYZ(tex, make_float3(position.x, position.y - h.y / 2.0f, position.z));
	dV_dZ -= this->ValueAtXYZ(tex, make_float3(position.x, position.y, position.z - h.z / 2.0f));

	return { dV_dX / h.x ,dV_dY / h.y, dV_dZ / h.z };
}



__device__ float IsosurfaceHelper::Velocity_Magnitude::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 _velocity = tex3D<float4>(tex, position.x, position.y, position.z);
	float3 velocity = make_float3(_velocity.x, _velocity.y, _velocity.z);
	return fabsf(sqrtf(dot(velocity, velocity)));
}

__device__ float IsosurfaceHelper::Velocity_X::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z).x;
}

__device__ float IsosurfaceHelper::Velocity_Y::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).y;
}

__device__ float IsosurfaceHelper::Velocity_Z::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).z;
}

__device__ float IsosurfaceHelper::ShearStress::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 dV_dY = tex3D<float4>(tex, position.x, position.y + 0.001 / 2.0f, position.z);
	
	dV_dY -= tex3D<float4>(tex, position.x, position.y - 0.001 / 2.0f, position.z);

	float2 ShearStress =make_float2(dV_dY.x / 0.001f, dV_dY.z / 0.001f);

	return fabsf(sqrtf(dot(ShearStress, ShearStress)));
}


__device__ float4 IsosurfaceHelper::Position::ValueAtXY(cudaTextureObject_t tex, float2 position)
{
	return   tex2D<float4>(tex, position.x, position.y);
}




__device__  float2 IsosurfaceHelper::Position::GradientAtXY_Grid(cudaSurfaceObject_t surf, int2 gridPosition)
{
	float dH_dX = this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x + 1, gridPosition.y));
	float dH_dY = this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x, gridPosition.y + 1));

	dH_dX -= this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x -1, gridPosition.y));
	dH_dY -= this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x, gridPosition.y -1));

	return make_float2(-dH_dX, -dH_dY);
}


__device__  float IsosurfaceHelper::Position::ValueAtXY_Surface_float(cudaSurfaceObject_t tex, int2 gridPos)
{
	float data;
	surf2Dread(&data, tex, gridPos.x *sizeof(float), gridPos.y);

	return data;
};

__device__  float4 IsosurfaceHelper::Position::ValueAtXY_Surface_float4(cudaSurfaceObject_t tex, int2 gridPos)
{
	float4 data;
	surf2Dread(&data, tex, gridPos.x * 4 * sizeof(float), gridPos.y);

	return data;
};




