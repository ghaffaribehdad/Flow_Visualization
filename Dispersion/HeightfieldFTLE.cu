#include "HightfieldFTLE.h"
#include "..//Cuda/helper_math.h"
#include "..//Particle/ParticleHelperFunctions.h"
#include "DispersionHelper.h"


__host__ bool HeightfieldFTLE::InitializeParticles()
{
	this->n_particles = dispersionOptions->gridSize_2D[0] * dispersionOptions->gridSize_2D[1] * FTLE_NEIGHBOR;
	this->h_particle = new Particle[n_particles];
	seedParticle_ZY_Plane_FTLE
	(
		h_particle,
		ARRAYTOFLOAT3(solverOptions->gridDiameter),
		ARRAYTOINT2(dispersionOptions->gridSize_2D),
		dispersionOptions->seedWallNormalDist,
		dispersionOptions->tilt_deg,
		dispersionOptions->ftleDistance
	);

	size_t Particles_byte = sizeof(Particle) * n_particles;

	gpuErrchk(cudaMalloc((void**)&this->d_particle, Particles_byte));
	gpuErrchk(cudaMemcpy(this->d_particle, this->h_particle, Particles_byte, cudaMemcpyHostToDevice));

	delete[] h_particle;

	return true;
}



void HeightfieldFTLE::trace3D_path_Single()
{
	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(n_particles);

	RK4STEP RK4Step = RK4STEP::ODD;

	for (int i = 0; i < solverOptions->lastIdx - solverOptions->firstIdx; i++)
	{
		if (i == 0)
		{
			// Load i 'dx field in volume_IO into field
			this->LoadVelocityfield(i + solverOptions->currentIdx);
			// Copy and initialize velocityfield texture
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
			// Release the velocityfield from host (volume_IO)
			primary_IO.release();

			// Same procedure for the second texture
			this->LoadVelocityfield(i + solverOptions->currentIdx + 1);
			this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
			primary_IO.release();

		}
		else
		{
			// Even integration steps
			if (i % 2 == 0)
			{

				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_1.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_1);
				primary_IO.release();

				RK4Step = RK4STEP::ODD;
			}
			// Odd integration steps
			else
			{
				this->LoadVelocityfield(i + solverOptions->currentIdx);
				this->velocityField_0.release();
				this->initializeVolumeTexuture(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap, velocityField_0);
				primary_IO.release();

				RK4Step = RK4STEP::EVEN;

			}

		}

		// initialize proper velocityfield

		// trace
		traceDispersion3D_path << < blocks, thread >> >
			(
				d_particle,
				s_HeightSurface_Primary.getSurfaceObject(),
				s_HeightSurface_Primary_Ex.getSurfaceObject(),
				this->velocityField_0.getTexture(),
				this->velocityField_1.getTexture(),
				*solverOptions,
				*dispersionOptions,
				RK4Step,
				i
				);
	}
	// Calculates the gradients and store it in the cuda surface
	cudaFree(d_particle);
}