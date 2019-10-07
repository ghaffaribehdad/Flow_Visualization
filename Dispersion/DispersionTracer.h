#pragma once
#include <string>
#include "..//Options/SolverOptions.h"
#include "..//Options/DispresionOptions.h"
#include "..//Particle/Particle.h"
#include "..//Volume/Volume_IO.h"
#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Cuda/cudaSurface.h"
#include "..//Cuda/CudaArray.h"

class DispersionTracer
{
public:

	bool initialize();
	void setResources(SolverOptions* _solverOption, DispersionOptions* _dispersionOptions);
	void release();
	void trace();
	//float * read(); // shouldn't we use the a unified reader for the visualization? (pathline is tricky then!)


private:
	bool InitializeVelocityField(int ID);
	bool InitializeParticles();
	bool InitializeHeightArray();

	DispersionOptions* dispersionOptions;
	SolverOptions* solverOptions;

	Volume_IO volume_IO;
	VolumeTexture volumeTexture;

	Particle* h_particle;
	Particle* d_particle;
	
	unsigned int n_particles;
	
	float* field;						// a pointer to the velocity field
	CudaSurface heightSurface;			// cuda surface storing the results
	CudaArray_3D<float4> heightArray;

	bool read();		
};



