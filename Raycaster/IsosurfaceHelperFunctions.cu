
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


__device__ float4 IsosurfaceHelper::Position::ValueAtXY(cudaTextureObject_t tex, float3 position)
{
	return   tex2D<float4>(tex, position.x, position.z);
}


// position is normalized ->[0,1]
__device__  float3 IsosurfaceHelper::Position::GradientAtXY_Height(cudaTextureObject_t tex, float3 position, int2 gridSize)
{
	// Interpolated gradient
	float3 result = { 0.0f, 0.0f, 0.0f };
	// Mesh size in XZ plane
	float3 h = make_float3(1.0f / gridSize.x, 0.0f, 1.0 / gridSize.y);

	float3* edges = new float3[4];

	float2 gridPos_floor	= make_float2(h.x,h.z)	*  make_int2(floor(position.x / h.x), floor(position.z / h.z));
	float2 gridPos_ceil		= make_float2(h.x, h.z) *  make_int2(ceil(position.x / h.x), ceil(position.z / h.z));
	
	edges[0] = make_float3(gridPos_floor.x, 0, gridPos_floor.y);
	edges[1] = make_float3(gridPos_floor.x, 0, gridPos_ceil.y);
	edges[2] = make_float3(gridPos_ceil.x, 0, gridPos_floor.y);
	edges[3] = make_float3(gridPos_ceil.x, 0, gridPos_ceil.y);
	
	result	 = GradientAtXY_Height_Grid(tex, edges[0], gridSize) *(	position.x - edges[0].x	)*(position.z - edges[0].z) / (h.x * h.z);
	result	+= GradientAtXY_Height_Grid(tex, edges[0], gridSize) *(	position.x - edges[1].x	)*(edges[1].z - position.z) / (h.x * h.z);
	result	+= GradientAtXY_Height_Grid(tex, edges[0], gridSize) *(	edges[2].x - position.x )*(position.z - edges[2].z) / (h.x * h.z);
	result	+= GradientAtXY_Height_Grid(tex, edges[0], gridSize) *( edges[3].x - position.x )*(edges[3].z - position.z) / (h.x * h.z);

	delete[] edges;

	return result;


}




__device__  float3 IsosurfaceHelper::Position::GradientAtXY_Height_Grid(cudaTextureObject_t tex, float3 position, int2 gridSize)
{

	float2 h = { 1.0f, 1.0f};
	h = h / gridSize;


	float dV_dX = this->ValueAtXY(tex, make_float3(position.x + h.x / 2.0f, position.y, position.z)).y;
	float dV_dY = this->ValueAtXY(tex, make_float3(position.x, position.y + h.y / 2.0f, position.z)).y;

	dV_dX -= this->ValueAtXY(tex, make_float3(position.x - h.x / 2.0f, position.y, position.z)).y;
	dV_dY -= this->ValueAtXY(tex, make_float3(position.x, position.y - h.y / 2.0f, position.z)).y;

	return {dV_dX / h.x ,-1, dV_dY / h.y };
}
