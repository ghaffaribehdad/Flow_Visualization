#pragma once
#include "Particle.h"
#include "../Options/SolverOptions.h"

//// Initialize the position of the particle on a plane in ZY direction
//void seedParticle_ZY_Plane
//(
//	Particle* particle,
//	float* gridDiameter,
//	const int* gridSize,
//	const float& y_slice
//);

// Initialize the position of the particle on a tilted plane in ZY direction
void seedParticle_tiltedPlane
(
	Particle* particle,
	const float3& gridDiameter,
	const int2& gridSize,
	const float& y_slice,
	const float& tilt
); 

// At each point 7 particle are seeded for 3D FTLE Calculation
void seedParticle_ZY_Plane_FTLE
(
	Particle* particle,
	const float3& gridDiameter,
	const int2& gridSize,
	const float& y_slice,
	const float& tilt,
	const float& delta_x
); 



void seedParticleGridPoints(Particle* particle, const SolverOptions* solverOptions);

void  seedParticleRandom(Particle* particle, const SolverOptions* solverOptions);