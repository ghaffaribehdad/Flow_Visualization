#pragma once

#include "cuda_runtime_api.h"
#include "BoundingBox.h"
#include <DirectXMath.h>
#include "../Options/RaycastingOptions.h"
#include "../Cuda/helper_math.h"
#include "../Cuda/CudaHelperFunctions.h"
#include "IsosurfaceHelperFunctions.h"
#include "CallerFunctions.h"

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
	RaycastingOptions raycastingOptions
	);




__global__ void CudaIsoSurfacRenderer_Double
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions
);


__global__ void CudaIsoSurfacRenderer_Multiscale
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	cudaTextureObject_t field2,
	int rays,
	RaycastingOptions raycastingOptions
);



template <typename Observable>
__global__ void CudaIsoSurfacRenderer_float_PlaneColor
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	int3 gridSize,
	RaycastingOptions raycastingOptions,
	SolverOptions solverOptions
)
{

	Observable observable;
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);

		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = 0.1f;
			float f = 1000.0f;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;

			for (float t = NearFar.x; t < NearFar.y; t = t + raycastingOptions.samplingRate_0)
			{


				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture



				switch (raycastingOptions.projectionPlane)
				{

				case(IsoMeasure::ProjectionPlane::ZXPLANE):
				{
					if (
						position.y < d_boundingBox.m_dimensions.y * raycastingOptions.wallNormalClipping &&
						position.y > d_boundingBox.m_dimensions.y * raycastingOptions.wallNormalClipping - raycastingOptions.planeThinkness
						)
					{

						float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, gridSize);
						float value = observable.ValueAtXYZ_Tex(field1, relativePos);
						float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

						float3 rgb_min =
						{
							raycastingOptions.minColor[0],
							raycastingOptions.minColor[1],
							raycastingOptions.minColor[2],
						};

						float3 rgb_max =
						{
							raycastingOptions.maxColor[0],
							raycastingOptions.maxColor[1],
							raycastingOptions.maxColor[2],
						};

						float y_saturated = 0.0f;
						float3 rgb = { 0,0,0 };
						float3 rgb_complement = { 0,0,0 };


						y_saturated = (value - raycastingOptions.minVal) / (raycastingOptions.maxVal - raycastingOptions.minVal);
						y_saturated = saturate(y_saturated);

						if (y_saturated > 0.5f)
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_max;
							rgb = rgb_complement * (1 - 2 * (y_saturated - 0.5)) + rgb_max;
						}
						else
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_min;
							rgb = rgb_complement * (2 * y_saturated) + rgb_min;
						}



						float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

						// write back color and depth into the texture (surface)
						// stride size of 4 * floats for each texel
						surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
						break;
					}
					break;
				}

				case(IsoMeasure::ProjectionPlane::XYPLANE):
				{

					if (
						position.z < d_boundingBox.m_dimensions.z * raycastingOptions.wallNormalClipping &&
						position.z > d_boundingBox.m_dimensions.z * raycastingOptions.wallNormalClipping - raycastingOptions.planeThinkness
						)
					{

						float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, gridSize);

						float value = observable.ValueAtXYZ_Tex(field1, relativePos);
						float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

						float3 rgb_min =
						{
							raycastingOptions.minColor[0],
							raycastingOptions.minColor[1],
							raycastingOptions.minColor[2],
						};

						float3 rgb_max =
						{
							raycastingOptions.maxColor[0],
							raycastingOptions.maxColor[1],
							raycastingOptions.maxColor[2],
						};

						float y_saturated = 0.0f;
						float3 rgb = { 0,0,0 };
						float3 rgb_complement = { 0,0,0 };


						y_saturated = (value - raycastingOptions.minVal) / (raycastingOptions.maxVal - raycastingOptions.minVal);
						y_saturated = saturate(y_saturated);

						if (y_saturated > 0.5f)
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_max;
							rgb = rgb_complement * (1 - 2 * (y_saturated - 0.5)) + rgb_max;
						}
						else
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_min;
							rgb = rgb_complement * (2 * y_saturated) + rgb_min;
						}



						float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

						// write back color and depth into the texture (surface)
						// stride size of 4 * floats for each texel
						surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
						break;
					}



					break;
				}

				case(IsoMeasure::ProjectionPlane::YZPLANE):
				{

					float planePos = d_boundingBox.m_dimensions.x * raycastingOptions.wallNormalClipping;
					if (solverOptions.projection == Projection::STREAK_PROJECTION)
					{
						planePos = planePos - (solverOptions.timeDim / 2) + (solverOptions.currentIdx - solverOptions.firstIdx) *(solverOptions.timeDim / (solverOptions.lastIdx - solverOptions.firstIdx));
					}
					if (
						position.x < planePos &&
						position.x > planePos - raycastingOptions.planeThinkness
						)
					{

						float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

						if (solverOptions.projection == Projection::STREAK_PROJECTION)
						{
							position.x = position.x + (solverOptions.timeDim / 2) - (solverOptions.currentIdx - solverOptions.firstIdx) *(solverOptions.timeDim / (solverOptions.lastIdx - solverOptions.firstIdx));
						}

						float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, gridSize);

						float value = observable.ValueAtXYZ_Tex(field1, relativePos);


						float3 rgb_min =
						{
							raycastingOptions.minColor[0],
							raycastingOptions.minColor[1],
							raycastingOptions.minColor[2],
						};

						float3 rgb_max =
						{
							raycastingOptions.maxColor[0],
							raycastingOptions.maxColor[1],
							raycastingOptions.maxColor[2],
						};

						float y_saturated = 0.0f;
						float3 rgb = { 0,0,0 };
						float3 rgb_complement = { 0,0,0 };


						y_saturated = (value - raycastingOptions.minVal) / (raycastingOptions.maxVal - raycastingOptions.minVal);
						y_saturated = saturate(y_saturated);

						if (y_saturated > 0.5f)
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_max;
							rgb = rgb_complement * (1 - 2 * (y_saturated - 0.5)) + rgb_max;
						}
						else
						{
							rgb_complement = make_float3(1, 1, 1) - rgb_min;
							rgb = rgb_complement * (2 * y_saturated) + rgb_min;
						}



						float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

						// write back color and depth into the texture (surface)
						// stride size of 4 * floats for each texel
						surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
						break;
					}



					break;
				}

				}




				// check if we have a hit 



			}


		}

	}


}