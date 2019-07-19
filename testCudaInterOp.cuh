#include "cuda_runtime.h"

__global__ void testSurface(cudaSurfaceObject_t surfaceObject);
__host__ void check(cudaSurfaceObject_t  surfaceObject);
