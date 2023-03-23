#pragma once
#include "CallerFunctions.h"
#include "IsosurfaceHelperFunctions.h"


__device__ float callerValueAtTex(int channel, cudaTextureObject_t tex, float3 position, float3  gridDiameter, int3 gridSize) {
	switch (channel)
	{
	case IsoMeasure::VELOCITY_X:
		return Channel_X::ValueAtXYZ(tex, position);
		break;

	case IsoMeasure::VELOCITY_Y:
		return Channel_Y::ValueAtXYZ(tex, position);
		break;

	case IsoMeasure::VELOCITY_Z:
		return Channel_Z::ValueAtXYZ(tex, position);
		break;

	case IsoMeasure::VELOCITY_W:
		return Channel_W::ValueAtXYZ(tex, position);
		break;

	case IsoMeasure::VELOCITY_MAGNITUDE:
		return Velocity_Magnitude::ValueAtXYZ(tex, position);
		break;

	case IsoMeasure::SHEAR_STRESS:
		return ShearStress::ValueAtXYZ(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::KINETIC_ENERGY:
		return KineticEnergy::ValueAtXYZ(tex, position);
		break;
		
	case IsoMeasure::LAMBDA2:
		return Lambda2::ValueAtXYZ(tex, position, gridDiameter, gridSize);
		break;
	}

	return 0;
}

__device__ float callerValueAtTex(int channel, cudaTextureObject_t tex, float3 position, float3  gridDiameter , int3 gridSize, float sigma , int3 offset0, int3 offset1)
{
	switch (channel)
	{
	case IsoMeasure::Finite_TIME:
		return FTLE::ValueAtXYZ(tex, position, gridDiameter, gridSize,sigma,offset0,offset1);
		break;
	}

	return 0;
}

__device__ float3 callerGradientAtTex(int measure, cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize, const float & sigma, const int3 & offset0, const int3 & offset1)
{
	switch (measure)
	{
	case IsoMeasure::Finite_TIME:
		return FTLE::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize, sigma, offset0, offset1);
	}
	return { 0,0,0 };
}

__device__ float3 callerGradientAtTex(int i, cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
{
	switch (i)
	{
	case IsoMeasure::VELOCITY_X:
		return Channel_X::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::VELOCITY_Y:
		return Channel_Y::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::VELOCITY_Z:
		return Channel_Z::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::VELOCITY_W:
		return Channel_W::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::VELOCITY_MAGNITUDE:
		return Velocity_Magnitude::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::SHEAR_STRESS:
		return ShearStress::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::KINETIC_ENERGY:
		return KineticEnergy::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;

	case IsoMeasure::LAMBDA2:
		return Lambda2::GradientAtXYZ_Tex(tex, position, gridDiameter, gridSize);
		break;
	}
	return { 0,0,0 };
}

__device__ float3 callerGradientAtTex(int i, cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
{
	switch (i)
	{
	case IsoMeasure::DIFFERENCE_X:
		return Difference_X::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::DIFFERENCE_Y:
		return Difference_Y::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::DIFFERENCE_Z:
		return Difference_Z::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::DIFFERENCE_W:
		return Difference_W::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::AVERAGE_DOUBLE:
		return Average_x_double::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::MAXIMUM_DOUBLE:
		return Max_x_double::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::MINIMUM_DOUBLE:
		return Min_x_double::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	case IsoMeasure::NORMAL_CURVES:
		return normalCurve_x_double::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;	
	case IsoMeasure::IMPORTANCE_DIFF_LESS:
		return normalCurve_x_double::GradientAtXYZ_Tex_double(tex0, tex1, position, gridDiameter, gridSize);
		break;
	
	default:
		return callerGradientAtTex(i,tex0,position,gridDiameter,gridSize);
		break;
	}
}

__device__ float callerValueAtTex(int channel, cudaTextureObject_t tex0, cudaTextureObject_t tex1, float3 position, float3  gridDiameter, int3 gridSize)
{
	switch (channel)
	{
	case IsoMeasure::DIFFERENCE_X:
		return Difference_X::ValueAtXYZ_double(tex0,tex1, position);
		break;
	case IsoMeasure::DIFFERENCE_Y:
		return Difference_Y::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::DIFFERENCE_Z:
		return Difference_Z::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::DIFFERENCE_W:
		return Difference_W::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::AVERAGE_DOUBLE:
		return Average_x_double::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::MAXIMUM_DOUBLE:
		return Max_x_double::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::MINIMUM_DOUBLE:
		return Min_x_double::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::NORMAL_CURVES:
		return normalCurve_x_double::ValueAtXYZ_double(tex0, tex1, position);
		break;
	case IsoMeasure::IMPORTANCE_DIFF_LESS:
		return normalCurve_x_double::ValueAtXYZ_double(tex0, tex1, position);
		break;
	default:
		return callerValueAtTex(channel,tex0,position,gridDiameter,gridSize);
		break;
	}

}
