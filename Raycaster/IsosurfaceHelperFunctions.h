#pragma once
#include <cuda_runtime.h>

__device__ float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position);


namespace FetchTextureSurface
{
	
	struct Measure
	{
		//Return the texel at XYZ of the Texture (Boundaries are controlled by the cudaTextureAddressMode
		__device__	virtual	float		ValueAtXYZ_Tex		(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float		ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__  virtual	float3		GradientAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position, const float3 & gridDiameter, const int3 & gridSize);
		__device__  virtual	float3		GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);

		__device__	static	float3		ValueAtXYZ_XYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_XY_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_XZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_YZ_Tex(cudaTextureObject_t tex, const float3 & position);

		__device__	static	float3		ValueAtXYZ_XYZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_XY_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_XZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_YZ_Surf(cudaSurfaceObject_t surf, const int3 & position);

		


	};

	struct Channel_X : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_Y : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_Z : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_W : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) override;
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position) override;

	};


	struct Velocity_Magnitude : public Measure
	{
		// calculates the value of the field at position XYZ
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) override;
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position) override;


	};

	struct ShearStress : public Measure
	{
		// calculates the value of the field at position XYZ
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) override;


	};

	struct TurbulentDiffusivity : public Measure
	{
		__device__ float ValueAtXYZ_avgtemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
		__device__ float3 GradientAtGrid_AvgTemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
		__device__ float3 binarySearch_avgtemp(
			cudaTextureObject_t field,
			cudaTextureObject_t average_temp,
			int3& _gridSize,
			float3& _position,
			float3& gridDiameter,
			float3& _samplingStep,
			float& value,
			float& tolerance,
			int maxIteration
		);
	};

	struct Position : public Measure
	{
		//calculates the value of the field at position XYZ
		__device__ float4 ValueAtXY(cudaTextureObject_t tex, float2 position);
		__device__ float ValueAtXY_Surface_float(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t tex, int3 gridPos);
		__device__ float4 ValueAtXY_Surface_float4(cudaSurfaceObject_t tex, int2 gridPos);
		__device__ float2 GradientAtXY_Grid(cudaSurfaceObject_t surf, int2 gridPosition);
		__device__ float2 GradientAtXYZ_Grid(cudaSurfaceObject_t surf, int3 gridPosition);
		__device__ float2 GradientFluctuatuionAtXT(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize);
		__device__ float2 GradientFluctuatuionAtXZ(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize);

	};



}