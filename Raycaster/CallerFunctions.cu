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
