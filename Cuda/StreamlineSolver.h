#pragma once

#include "CudaSolver.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"


#include "device_launch_parameters.h"
#include "texture_fetch_functions.h"
#include <vector>
#include <stdio.h>
#include "../Particle/Particle.h"



class StreamlineSolver : public CUDASolver
{

public:

	__host__ bool solve() override;
	__host__ bool release() override;

	__host__ bool releaseVolumeIO();
	__host__ bool releaseVolumeTexture();
	__host__ bool loadVolumeTexture();
	
};







