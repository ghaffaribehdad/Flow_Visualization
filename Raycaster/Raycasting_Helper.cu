

#include "Raycasting_Helper.h"
#include "../Cuda/helper_math.h"
typedef unsigned char uchar;





















__global__ void CudaIsoSurfacRenderer_Single
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.m_dimensions / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		//float2 NearFar = findIntersections(pixelPos, d_boundingBox);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		//float4 rgba = { 1, 0,0, 1 };

		//// write back color and depth into the texture (surface)
		//// stride size of 4 * floats for each texel
		//surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
				float value = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				if (value > raycastingOptions.isoValue_0)
				{
					float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					}

					// calculates gradient
					float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_0) ^ raycastingOptions.brightness;
					float3 rgb = raycastingColor * diffuse;
					
					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}

				if (raycastingOptions.adaptiveSampling)
				{
					float gridStep = findExitPoint3D(position, dir, cellSize);
					t += gridStep;
				}
				else
				{
					t = t + raycastingOptions.samplingRate_0;
				}
			}


		}

	}


}


__global__ void CudaIsoSurfacRenderer_Double
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.m_dimensions / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = NearFar.x;

			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;
				float3 texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				if (callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.m_dimensions,d_boundingBox.gridSize) > raycastingOptions.isoValue_0)
				{

					//if (raycastingOptions.binarySearch)
					//{
					//	float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
					//	position = binarySearch(field0, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
					//	texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					//	//update t

					//}
					// calculates gradient
					float3 gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					float3 rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparecny + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparecny);


					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					
					while (t < NearFar.y && callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.m_dimensions, d_boundingBox.gridSize) > raycastingOptions.isoValue_0)
					{
						// Position of the isosurface
						float3 position = pixelPos + (rayDir * t);
						position += d_boundingBox.m_dimensions / 2.0;
						//Relative position calculates the position of the point on the CUDA texture
						texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

						// it true then we have the second hit
						if (callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, d_boundingBox.m_dimensions, d_boundingBox.gridSize) > raycastingOptions.isoValue_1)
						{

							if (raycastingOptions.binarySearch)
							{
								float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;
								position = binarySearch(field1, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
								texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
							}
							
							float3 gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

							// Now check for the angle between gradient and the ray
							float angle = dot(gradient1, rayDir);
							if (angle < 0)
							{

								break;
							}
							else
							{
								float3 rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								float depth1 = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
								rgb = rgb1 * raycastingOptions.transparecny + (1 - raycastingOptions.transparecny)*rgb;
								rgba = { rgb.x, rgb.y, rgb.z, depth1 };
								break;
							}

						}

						if (raycastingOptions.adaptiveSampling)
						{
							t += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t = t + raycastingOptions.samplingRate_1;
						}
					}

					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}


				if (raycastingOptions.adaptiveSampling)
				{
					t += findExitPoint3D(position, dir, cellSize);;
				}
				else
				{
					t = t + raycastingOptions.samplingRate_0;
				}
			}


		}

	}


}

__global__ void CudaIsoSurfacRenderer_Double_Separate
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.m_dimensions / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		//float2 NearFar = findIntersections(pixelPos, d_boundingBox);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
				float value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
				float value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);

				if (value_0 > raycastingOptions.isoValue_0)
				{
					float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					}

					// calculates gradient
					float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_0) ^ raycastingOptions.brightness;
					float3 rgb = raycastingColor * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}


				if (value_1 > raycastingOptions.isoValue_1)
				{
					float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field1, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						relativePos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					}

					// calculates gradient
					float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, relativePos, d_boundingBox.m_dimensions, d_boundingBox.gridSize);


					// shading (no ambient)
					float diffuse = max(dot(normalize(gradient), d_boundingBox.m_viewDir), 0.0f);
					float3 raycastingColor = Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness;
					float3 rgb = raycastingColor * diffuse;

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}

				if (raycastingOptions.adaptiveSampling)
				{
					float gridStep = findExitPoint3D(position, dir, cellSize);
					t += gridStep;
				}
				else
				{
					t = t + raycastingOptions.samplingRate_1;
				}
			}


		}

	}


}

__global__ void CudaIsoSurfacRenderer_Multiscale
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	cudaTextureObject_t field1,
	cudaTextureObject_t field2,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions
)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
		float3 cellSize = d_boundingBox.m_dimensions / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);


		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			float t = NearFar.x;

			int3 gridSize = d_boundingBox.gridSize;
			int3 gridSize_L1 = gridSize / 2;
			int3 gridSize_L2 = gridSize_L1 / 2;

			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;
				float3 texPos = world2Tex(position, d_boundingBox.m_dimensions, gridSize);
				float3 texPos_L1 = world2Tex(position, d_boundingBox.m_dimensions, gridSize_L1);
				float3 texPos_L2 = world2Tex(position, d_boundingBox.m_dimensions, gridSize_L2);

				
				if (callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.m_dimensions, gridSize_L2) > raycastingOptions.isoValue_0)
				{
					// calculates gradient
					float3 gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.m_dimensions, gridSize_L2);
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					float3 rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparecny + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparecny);
					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					while (t < NearFar.y && callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2,d_boundingBox.m_dimensions,gridSize_L2) > raycastingOptions.isoValue_0)
					{

						
						// Position of the isosurface
						float3 position = pixelPos + (rayDir * t);
						position += d_boundingBox.m_dimensions / 2.0;
						//Relative position calculates the position of the point on the CUDA texture

						texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
						texPos_L1 = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize / 2);
						texPos_L2 = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize / 4);



						// it true then we have the second hit (L1)
						if (callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos_L1, d_boundingBox.m_dimensions, gridSize_L1) > raycastingOptions.isoValue_1)
						{

							float3 gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos_L1, d_boundingBox.m_dimensions, gridSize_L1);

							float angle = dot(gradient1, rayDir);
							if (angle < 0)
							{

								break;
							}
							else
							{
								float3 rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								float depth1 = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
								rgb = rgb1 * raycastingOptions.transparecny + (1 - raycastingOptions.transparecny)*rgb;
								rgba = { rgb.x, rgb.y, rgb.z, depth1 };
								break;
							}



							break;
						}

						if (raycastingOptions.adaptiveSampling)
						{
							t += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t = t + raycastingOptions.samplingRate_1;
						}
					}

					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}


				if (raycastingOptions.adaptiveSampling)
				{
					t += findExitPoint3D(position, dir, cellSize);;
				}
				else
				{
					t = t + raycastingOptions.samplingRate_0;
				}
			}


		}

	}


}



__global__ void CudaIsoSurfacRenderer_Planar
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field0,
	int rays,
	RaycastingOptions raycastingOptions,
	RenderingOptions renderingOptions)
{
	int index = CUDA_INDEX;

	if (index < rays)
	{

		// determine pixel position based on the index of the thread
		int2 pixel = index2pixel(index, d_boundingBox.m_width);

		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);

		// if inside the bounding box
		if (NearFar.y != -1 && NearFar.x != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.m_dimensions / 2.0;
			int3 gridSize = d_boundingBox.gridSize;


			for (float t = NearFar.x; t < NearFar.y; t = t + raycastingOptions.samplingRate_0)
			{


				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.m_dimensions / 2.0;

				bool hit = false;

				switch (raycastingOptions.projectionPlane)
				{

				case(IsoMeasure::ProjectionPlane::ZXPLANE):

					if (position.y < d_boundingBox.m_dimensions.y * raycastingOptions.planeProbePosition &&
						position.y > d_boundingBox.m_dimensions.y * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;
					break;
				case(IsoMeasure::ProjectionPlane::XYPLANE):

					if (position.z < d_boundingBox.m_dimensions.z * raycastingOptions.planeProbePosition &&
						position.z > d_boundingBox.m_dimensions.z * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;
					break;
				case(IsoMeasure::ProjectionPlane::YZPLANE):

					if (position.x < d_boundingBox.m_dimensions.x * raycastingOptions.planeProbePosition &&
						position.x > d_boundingBox.m_dimensions.x * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;

					break;
				}

				if (hit == true)
				{
					float3 relativePos = world2Tex(position, d_boundingBox.m_dimensions, gridSize);
					float value = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.m_dimensions, gridSize);
					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float3 rgb = { 0,0,0 };
					rgb = colorCode(raycastingOptions.minColor, raycastingOptions.maxColor, value, raycastingOptions.minVal, raycastingOptions.maxVal);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };
					
					// write back color and depth into the texture (surface)
					// stride size of 4 * floats for each texel
					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);

					break;
				}
			}
		}

	}
}



// return the far and near intersection with bounding box
__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox)
{

	bool hit = true;


	float arrayPixelPos[3] = { pixelPos.x, pixelPos.y, pixelPos.z };

	float tNear = -100000;
	float tFar = +100000;

	float3 dir = normalize(pixelPos - boundingBox.m_eyePos);
	float D[3] = { dir.x,dir.y,dir.z };

	// iterates over x,y,z planes
	for (int i = 0; i < 3; i++)
	{
		float plane1 = boundingBox.boxFaces[2 * i];
		float plane2 = boundingBox.boxFaces[2 * i + 1];
		float t1 = 0;
		float t2 = 0;


		// check if ray and axis are aligned
		if (D[i] == 0)
		{
			if (arrayPixelPos[i] < plane1 || arrayPixelPos[i] > plane2)
			{
				hit = false;
				break;
			}
		}
		else
		{
			t1 = (plane1 - arrayPixelPos[i]) / D[i];
			float tTemp = (plane2 - arrayPixelPos[i]) / D[i];

			// Sort t1 and t2
			if (t1 <= tTemp)
			{
				t2 = tTemp;
			}
			else
			{
				t2 = t1;
				t1 = tTemp;
			}
			

			if (t1 > tNear)
				tNear = t1;
			if (t2 < tFar)
				tFar = t2;
			if (tNear > tFar)
			{
				hit = false;
				break;
			}
			if (tFar < 0)
			{
				hit = false;
				break;
			}
		}

	}

	if (hit)
	{
		if (tNear < 0)
		{
			return { 0,tFar };
		}
		else
		{
			return { tNear,tFar };
		}
	}
	else
	{
		return { -1,-1 };
	}

}


// return the far and near intersection with bounding box
__device__ float2 findEnterExit(const float3 & pixelPos, const float3  & dir, float boxFaces[6])
{

	bool hit = true;


	float arrayPixelPos[3] = { pixelPos.x, pixelPos.y, pixelPos.z };

	float tNear = -1000000000;
	float tFar = +1000000000;

	double D[3] = { dir.x,dir.y,dir.z };

	// iterates over x,y,z planes
	for (int i = 0; i < 3; i++)
	{
		double plane1 = boxFaces[2 * i];
		double plane2 = boxFaces[2 * i + 1];
		double t1 = 0;
		double t2 = 0;


		// check if ray and axis are aligned
		if (D[i] == 0)
		{
			if (arrayPixelPos[i] < plane1 || arrayPixelPos[i] > plane2)
			{
				hit = false;
				break;
			}
		}
		else
		{
			t1 = (plane1 - arrayPixelPos[i]) / D[i];
			float tTemp = (plane2 - arrayPixelPos[i]) / D[i];

			// Sort t1 and t2
			if (t1 <= tTemp)
			{
				t2 = tTemp;
			}
			else
			{
				t2 = t1;
				t1 = tTemp;
			}


			if (t1 > tNear)
				tNear = t1;
			if (t2 < tFar)
				tFar = t2;
			if (tNear > tFar)
			{
				hit = false;
				break;
			}
			if (tFar < 0)
			{
				hit = false;
				break;
			}
		}

	}

	if (hit)
	{
		if (tNear < 0)
		{
			return { 0,tFar };
		}
		else
		{
			return { tNear,tFar };
		}
	}
	else
	{
		return { -1,-1 };
	}

}



__device__ float findExitPoint(const float2& entery, const float2& dir, const float2 & cellSize)
{

	// First we find intersection on X and Then Y
	// Then we compare the ray parameter (t) and then we choose the minimum t
	
	float2 step = entery / cellSize;
	float2 edge = { 0,0 };

	if (dir.x < 0)
	{
		edge.x = (ceil(step.x) - 1) * cellSize.x;
	}
	else
	{
		edge.x = (floor(step.x) + 1) * cellSize.x;
	}

	if (dir.y < 0)
	{
		edge.y = (ceil(step.y) - 1) * cellSize.y;
	}
	else
	{
		edge.y = (floor(step.y) + 1) * cellSize.y;
	}

	float t_x = 0;
	float t_y = 0;


	t_x = (edge.x - entery.x) / dir.x;
	t_y = (edge.y - entery.y) / dir.y;

	// take the minimum value of ray parameter
	return fmax(fmin(t_x, t_y), 0.00001f);
}


__device__ float findExitPoint3D(const float3& entery, const float3& dir, const float3 & cellSize)
{

	float3 step = entery / cellSize;
	float3 edge = { 0,0,0 };

	

	if (dir.x < 0)
	{
		edge.x = (ceil(step.x) - 1) * cellSize.x;
	}
	else
	{
		edge.x = (floor(step.x) + 1) * cellSize.x;
	}

	if (dir.y < 0)
	{
		edge.y = (ceil(step.y) - 1) * cellSize.y;
	}
	else
	{
		edge.y = (floor(step.y) + 1) * cellSize.y;
	}

	if (dir.z < 0)
	{
		edge.z = (ceil(step.z) - 1) * cellSize.z;
	}
	else
	{
		edge.z = (floor(step.z) + 1) * cellSize.z;
	}

	float t_x = 0;
	float t_y = 0;
	float t_z = 0;


	t_x = (edge.x - entery.x) / dir.x;
	t_y = (edge.y - entery.y) / dir.y;
	t_z = (edge.z - entery.z) / dir.z;

	float t_min = t_x;
	if (t_y < t_min)
		t_min = t_y;
	if (t_z < t_min)
		t_min = t_z;
	// take the minimum value of ray parameter
	return fmax(t_min, 0.000001f);
}





__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j)
{
	// Height of the Image Plane
	float H = static_cast<float>(tan(boundingBox.FOV / 2.0) * 2.0 * boundingBox.distImagePlane);
	// Width of the Image Plane
	float W = H * boundingBox.aspectRatio;

	// Center of Image Plane
	float3 centerPos = boundingBox.m_eyePos + (boundingBox.nuv[0] * boundingBox.distImagePlane);

	// Left Corner of Image Plane
	float3 leftCornerPos = (centerPos + (boundingBox.nuv[1] * W / 2.0f) - (boundingBox.nuv[2] * H / 2.0f));

	float3 pixelPos = leftCornerPos - (boundingBox.nuv[1] * float(i) * W / float(boundingBox.m_width));
	pixelPos += boundingBox.nuv[2] * float(j) * H / float(boundingBox.m_height);

	return pixelPos;
}

__device__ uchar4 rgbaFloatToUChar(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return make_uchar4(uchar(rgba.x * 255.0f), uchar(rgba.y * 255.0f), uchar(rgba.z * 255.0f), uchar(rgba.w * 255.0f));
}    



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
)
{
	float3 position = _position;
	float3 relative_position = world2Tex(position, gridDiameter, gridSize);
	float3 samplingStep = samplingRate * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;

	while (fabsf(callerValueAtTex(isomeasure, field, relative_position, gridDiameter, gridSize) - value) > tolerance && counter < maxIteration)
	{

		if (callerValueAtTex(isomeasure, field, relative_position, gridDiameter, gridSize) - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			position = position - samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			position = position + samplingStep;
			relative_position = world2Tex(position, gridDiameter, gridSize);
			side = 1;

		}
		counter++;

	}

	return position;
}