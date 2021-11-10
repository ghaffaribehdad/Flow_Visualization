#pragma once

#include "cuda_runtime_api.h"
#include "BoundingBox.h"
#include <DirectXMath.h>
#include "../Options/RaycastingOptions.h"
#include "../Cuda/helper_math.h"
#include "../Cuda/CudaHelperFunctions.h"
#include "IsosurfaceHelperFunctions.h"
#include "CallerFunctions.h"
#include "../Options/RenderingOptions.h"

extern __constant__	BoundingBox d_boundingBox; // constant memory variable
extern __constant__	BoundingBox d_boundingBox_spacetime;


__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox);
__device__ float2 findEnterExit(const float3 & pixelPos, const float3  & dir, float boxFaces[6]);
__device__ float findExitPoint(const float2 & entery, const float2 & dir, const float2 & cellSize);
__device__ float findExitPoint3D(const float3 & entery, const float3 & dir, const float3 & cellSize);
__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j);
__device__ uchar4 rgbaFloatToUChar(float4 rgba);




__global__ void CudaIsoSurfacRenderer_Single
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
	);




__global__ void CudaIsoSurfacRenderer_Double
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Double_Separate
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);


__global__ void CudaIsoSurfacRenderer_Multiscale
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	cudaTextureObject_t field2,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);



__global__ void CudaIsoSurfacRenderer_Planar
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);



__device__ float3 binarySearch
(
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	int3& gridSize,
	float& value,
	float3& samplingRate,
	int & isomeasure,
	float & tolerance,
	int & maxIteration
);

