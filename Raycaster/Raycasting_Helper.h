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
#include "../Options/VisitationOptions.h"

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

__global__ void CudaDVR_VISITATION
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions,
	VisitationOptions	visitationOptions
);

__global__ void CudaIsoSurfacRenderer_Single_ColorCoded
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Projection
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);


__global__ void CudaIsoSurfacRenderer_Projection_Forward
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Projection_Backward
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Projection_Average
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Projection_Length
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

__global__ void CudaIsoSurfacRenderer_Multiscale_Triple
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	cudaTextureObject_t field2,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);


__global__ void CudaIsoSurfacRenderer_Multiscale_Defect
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	cudaTextureObject_t field2,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Double_Advanced
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

//__global__ void CudaIsoSurfacRenderer_Double_Transparency
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t field0,
//	cudaTextureObject_t field1,
//	int rays,
//	RaycastingOptions raycastingOptions,
//	RenderingOptions renderingOptions
//);

__global__ void CudaIsoSurfacRenderer_Double_Transparency_noglass
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);

__global__ void CudaIsoSurfacRenderer_Double_Transparency_noglass_multiLevel
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
);




__device__ float callerValueAtTexLOD
(
	int isoMeasure,
	cudaTextureObject_t & field_0, cudaTextureObject_t & field_1, cudaTextureObject_t & field_2,
	float3 & texPos_L0, float3 & texPos_L1, float3 & texPos_L2,
	float3 & gridDiameter,
	int3 & gridSize_L0, int3 & gridSize_L1, int3 & gridSize_L2,
	int level
);

__device__ float3 callerGradientAtTexLOD
(
	int &isoMeasure,
	cudaTextureObject_t & field_0, cudaTextureObject_t & field_1, cudaTextureObject_t & field_2,
	float3 & texPos_L0, float3 & texPos_L1, float3 & texPos_L2,
	float3 & gridDiameter,
	int3 & gridSize_L0, int3 & gridSize_L1, int3 & gridSize_L2,
	int & level
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


__device__ float3 gradientTrilinear(int isoMeasure, float3 texPos, cudaTextureObject_t field0, int3 & gridSize, float3 & gridDiameter);

__device__ float distanceToNormalCurves(
	int & isoMeasure,
	float & isoValue,
	float3 position,
	cudaTextureObject_t field0,
	int3 gridSize,
	float3 gridDiameter,
	float samplingStep
);


__global__ void visitationMapGenerator
(
	SolverOptions solverOptions,
	RaycastingOptions raycastingOptions,
	VisitationOptions visitationOptions,
	cudaTextureObject_t t_velocityField,
	cudaSurfaceObject_t	s_textureMap
);

__global__ void visitationMap_Diff
(
	SolverOptions solverOptions,
	RaycastingOptions raycastingOptions,
	VisitationOptions visitationOptions,
	cudaTextureObject_t t_velocityField,
	cudaSurfaceObject_t	s_textureMap
);