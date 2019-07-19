#include "testCudaInterOp.cuh"
#include "cuda_fp16.h"

__global__ void testSurface(cudaSurfaceObject_t surfaceObject)
{
	for (int i = 200; i < 700; i++)
		for (int j = 200; j < 700; j++)
		{
			{
				float Texel;
				float Texel2 = 222;
				surf2Dread(&Texel, surfaceObject, i, j);
				surf2Dwrite(Texel2, surfaceObject, i, j);
				surf2Dread(&Texel, surfaceObject, i, j);
			}

			
		}

}


__host__ void check(cudaSurfaceObject_t surfaceObject)
{
	testSurface << <100, 100 >> > (surfaceObject);
}