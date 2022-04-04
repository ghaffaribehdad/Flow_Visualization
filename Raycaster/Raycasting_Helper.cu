

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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);



		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);

			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += gridDiameter / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 texPos = world2Tex(position, gridDiameter, gridSize);
				float value = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);



				if (value > raycastingOptions.isoValue_0)
				{
					float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, d_boundingBox.gridDiameter, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
					}




					// calculates gradient
					float3 gradient = gradientTrilinear(raycastingOptions.isoMeasure_0,texPos,field0,gridSize,gridDiameter);
					//float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos,gridDiameter, gridSize);


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



__global__ void CudaIsoSurfacRenderer_Projection
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		float3 position;

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;
	

			// Projection of isosurfaces
			float t_x = -gridDiameter.x / 2.0;
			t_x = t_x + (12.0f * gridDiameter.x / (gridSize.x - 1));
			t_x = t_x - pixelPos.x;
			t_x = t_x / rayDir.x;

			if (t_x >= NearFar.x && t_x <= NearFar.y)
			{
				position = pixelPos + (rayDir * t_x);
				position = position + gridDiameter / 2.0;


				float3 texPos = world2Tex(position, gridDiameter, gridSize);

				int nStep_x = 0;
				float distance = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x + raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value < raycastingOptions.isoValue_0)
					{
						break;
					}

					nStep_x++;

					if (texPos.x + raycastingOptions.samplingRate_projection * nStep_x > gridSize.x)
					{
						break;
					}


				}

				distance = nStep_x * raycastingOptions.samplingRate_projection;
				float3 rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);
				float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

				float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);


			}

		}



	}

}




__global__ void CudaIsoSurfacRenderer_Projection_Backward
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
			float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
			float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
			float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

			float3 position;

			// if inside the bounding box
			if (NearFar.y != -1)
			{

				float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
				float3 gridDiameter = d_boundingBox.gridDiameter;
				int3 gridSize = d_boundingBox.gridSize;

				// near and far plane
				float n = renderingOptions.nearField;
				float f = renderingOptions.farField;

				// Add the offset to the eye position
				float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;


				// Projection of isosurfaces
				float t_x = -gridDiameter.x / 2.0;
				t_x = t_x + (raycastingOptions.projectionPlanePos * gridDiameter.x / (gridSize.x - 1));
				t_x = t_x - pixelPos.x;
				t_x = t_x / rayDir.x;

				if (t_x >= NearFar.x && t_x <= NearFar.y)
				{
					position = pixelPos + (rayDir * t_x);
					position = position + gridDiameter / 2.0;


					float3 texPos = world2Tex(position, gridDiameter, gridSize);

					int nStep_x = 0;
					float distance = 0;

					while (true)
					{
						float value = callerValueAtTex(
							raycastingOptions.isoMeasure_0,
							field0,
							make_float3(texPos.x - raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
							gridDiameter,
							gridSize
						);

						if (value < raycastingOptions.isoValue_0)
						{
							break;
						}


						nStep_x++;

						if (texPos.x - raycastingOptions.samplingRate_projection * nStep_x < 0)
						{
							break;
						}


					}

					distance = nStep_x * raycastingOptions.samplingRate_projection;
					float3 rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);

					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

					float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
					surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);


				}

			}



		}

}



__global__ void CudaIsoSurfacRenderer_Projection_Average
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		float3 position;

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;


			// Projection of isosurfaces
			float t_x = -gridDiameter.x / 2.0;
			t_x = t_x + (raycastingOptions.projectionPlanePos * gridDiameter.x / (gridSize.x - 1));
			t_x = t_x - pixelPos.x;
			t_x = t_x / rayDir.x;

			if (t_x >= NearFar.x && t_x <= NearFar.y)
			{
				position = pixelPos + (rayDir * t_x);
				position += gridDiameter / 2.0;


				float3 texPos = world2Tex(position, gridDiameter, gridSize);

				int nStep_x = 0;
				float distance_forward = 0;
				float distance_backward = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x - raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value < raycastingOptions.isoValue_0)
					{
						break;
					}


					nStep_x++;

					if (texPos.x - raycastingOptions.samplingRate_projection * nStep_x < 0)
					{
						break;
					}


				}

				distance_backward = nStep_x * raycastingOptions.samplingRate_projection;
				nStep_x = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x + raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value < raycastingOptions.isoValue_0)
					{
						break;
					}

					nStep_x++;

					if (texPos.x + raycastingOptions.samplingRate_projection * nStep_x > gridSize.x)
					{
						break;
					}


				}
				distance_forward = nStep_x * raycastingOptions.samplingRate_projection;

				float distance = 0.5f * (distance_backward + distance_forward);

				float3 rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);
				float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

				float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);


			}

		}



	}

}



__global__ void CudaIsoSurfacRenderer_Projection_Length
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		float3 position;

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

		

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;


			// Projection of isosurfaces
			float t_x = -gridDiameter.x / 2.0;
			t_x = t_x + (raycastingOptions.projectionPlanePos * gridDiameter.x / (gridSize.x - 1));
			t_x = t_x - pixelPos.x;
			t_x = t_x / rayDir.x;

			if (t_x >= NearFar.x && t_x <= NearFar.y)
			{
				position = pixelPos + (rayDir * t_x);
				position += gridDiameter / 2.0;


				float3 texPos = world2Tex(position, gridDiameter, gridSize);

				int nStep_x = 0;
				float distance_forward = 0;
				float distance_backward = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x - raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value > raycastingOptions.isoValue_0)
					{
						distance_backward++;
					}
					nStep_x++;

					if (texPos.x - raycastingOptions.samplingRate_projection * nStep_x < 0)
					{
						break;
					}


				}

				distance_backward = distance_backward * raycastingOptions.samplingRate_projection;
				nStep_x = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x + raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value > raycastingOptions.isoValue_0)
					{
						distance_forward++;
					}

					nStep_x++;

					if (texPos.x + raycastingOptions.samplingRate_projection * nStep_x > gridSize.x)
					{
						break;
					}


				}
				distance_forward = distance_forward * raycastingOptions.samplingRate_projection;

				float distance = distance_backward + distance_forward;

				float3 rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);
				float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

				float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);


			}

		}



	}

}



__global__ void CudaIsoSurfacRenderer_Projection_Forward
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);

		float3 position;

		// if inside the bounding box
		if (NearFar.y != -1)
		{

			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
			float3 gridDiameter = d_boundingBox.gridDiameter;
			int3 gridSize = d_boundingBox.gridSize;

			// near and far plane
			float n = renderingOptions.nearField;
			float f = renderingOptions.farField;

			// Add the offset to the eye position
			float3 eyePos = d_boundingBox.m_eyePos + gridDiameter / 2.0;


			// Projection of isosurfaces
			float t_x = -gridDiameter.x / 2.0;
			t_x = t_x + (raycastingOptions.projectionPlanePos * gridDiameter.x / (gridSize.x - 1));
			t_x = t_x - pixelPos.x;
			t_x = t_x / rayDir.x;

			if (t_x >= NearFar.x && t_x <= NearFar.y)
			{
				position = pixelPos + (rayDir * t_x);
				position += gridDiameter / 2.0;


				float3 texPos = world2Tex(position, gridDiameter, gridSize);

				int nStep_x = 0;
				float distance = 0;

				while (true)
				{
					float value = callerValueAtTex(
						raycastingOptions.isoMeasure_0,
						field0,
						make_float3(texPos.x + raycastingOptions.samplingRate_projection * nStep_x, texPos.y, texPos.z),
						gridDiameter,
						gridSize
					);

					if (value < raycastingOptions.isoValue_0)
					{
						break;
					}

					nStep_x++;

					if (texPos.x + raycastingOptions.samplingRate_projection * nStep_x > gridSize.x)
					{
						break;
					}


				}

				
				distance = nStep_x * raycastingOptions.samplingRate_projection;
				float3 rgb = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distance, renderingOptions.minMeasure, renderingOptions.maxMeasure);
				float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);

				float4 rgba = { rgb.x , rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, sizeof(float4) * pixel.x, pixel.y);


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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;

			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;
				float3 texPos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

				if (callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.gridDiameter,d_boundingBox.gridSize) > raycastingOptions.isoValue_0)
				{

					//if (raycastingOptions.binarySearch)
					//{
					//	float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
					//	position = binarySearch(field0, position, d_boundingBox.m_dimensions, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
					//	texPos = world2Tex(position, d_boundingBox.m_dimensions, d_boundingBox.gridSize);
					//	//update t

					//}
					// calculates gradient
					float3 gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					float3 rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparency_0 + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparency_0);


					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					
					while (t < NearFar.y && callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, d_boundingBox.gridDiameter, d_boundingBox.gridSize) > raycastingOptions.isoValue_0)
					{
						// Position of the isosurface
						float3 position = pixelPos + (rayDir * t);
						position += d_boundingBox.gridDiameter / 2.0;
						//Relative position calculates the position of the point on the CUDA texture
						texPos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

						// it true then we have the second hit
						if (callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, d_boundingBox.gridDiameter, d_boundingBox.gridSize) > raycastingOptions.isoValue_1)
						{

							if (raycastingOptions.binarySearch)
							{
								float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;
								position = binarySearch(field1, position, d_boundingBox.gridDiameter, d_boundingBox.gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
								texPos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
							}
							
							float3 gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

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
								rgb = rgb1 * raycastingOptions.transparency_0 + (1 - raycastingOptions.transparency_0)*rgb;
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;
			while (t < NearFar.y)
			{
				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;

				//Relative position calculates the position of the point on the CUDA texture
				float3 relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
				float value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
				float value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, relativePos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

				if (value_0 > raycastingOptions.isoValue_0)
				{
					float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, d_boundingBox.gridDiameter, d_boundingBox.gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
					}

					// calculates gradient
					float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);


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
						position = binarySearch(field1, position, d_boundingBox.gridDiameter, d_boundingBox.gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						relativePos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
					}

					// calculates gradient
					float3 gradient = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, relativePos, d_boundingBox.gridDiameter, d_boundingBox.gridSize);


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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;

			int3 gridSize_L0 = d_boundingBox.gridSize;
			int3 gridSize_L1 = gridSize_L0 / 2;
			int3 gridSize_L2 = gridSize_L1 / 2;
			float3 position;
			float value_hit0;
			float value_hit1;



			while (t < NearFar.y)
			{
				// Position of the isosurface
				position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;
				float3 texPos_L0 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L0);
				float3 texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L1);
				float3 texPos_L2 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L2);

				switch (raycastingOptions.fieldLevel_0)
				{
				case IsoMeasure::FieldLevel::L0:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
					break;
				case IsoMeasure::FieldLevel::L1:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
					break;
				case IsoMeasure::FieldLevel::L2:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
					break;
				}
				if (value_hit0 > raycastingOptions.isoValue_0)
				{
					float3 gradient0;

					switch (raycastingOptions.fieldLevel_0)
					{
					case IsoMeasure::FieldLevel::L0:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
						break;
					case IsoMeasure::FieldLevel::L1:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
						break;
					case IsoMeasure::FieldLevel::L2:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
						break;
					}
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					float3 rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparency_0 + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparency_0);
					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					while (t < NearFar.y && value_hit0 > raycastingOptions.isoValue_0)
					{


						// Position of the isosurface
						float3 position = pixelPos + (rayDir * t);
						position += d_boundingBox.gridDiameter / 2.0;
						//Relative position calculates the position of the point on the CUDA texture

						texPos_L0 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
						texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 2);
						texPos_L2 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 4);



						// it true then we have the second hit (L1)
						switch (raycastingOptions.fieldLevel_1)
						{
						case IsoMeasure::FieldLevel::L0:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
							break;
						case IsoMeasure::FieldLevel::L1:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
							break;
						case IsoMeasure::FieldLevel::L2:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
							break;
						}



						if (value_hit1 > raycastingOptions.isoValue_1)
						{

							float3 gradient1;

							switch (raycastingOptions.fieldLevel_1)
							{
							case IsoMeasure::FieldLevel::L0:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
								break;
							case IsoMeasure::FieldLevel::L1:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
								break;
							case IsoMeasure::FieldLevel::L2:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
								break;
							}


							float angle = dot(gradient1, rayDir);
							if (angle < 0)
							{

								break;
							}
							else
							{
								float3 rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								float depth1 = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
								rgb = rgb1 * raycastingOptions.transparency_0 + (1 - raycastingOptions.transparency_0)*rgb;
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

__global__ void CudaIsoSurfacRenderer_Multiscale_Triple
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;

			int3 gridSize_L0 = d_boundingBox.gridSize;
			int3 gridSize_L1 = gridSize_L0 / 2;
			int3 gridSize_L2 = gridSize_L1 / 2;
			float3 position;
			float value_hit0;
			float value_hit1;



			while (t < NearFar.y)
			{
				// Position of the isosurface
				position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;
				float3 texPos_L0 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L0);
				float3 texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L1);
				float3 texPos_L2 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L2);

				switch (raycastingOptions.fieldLevel_0)
				{
				case IsoMeasure::FieldLevel::L0:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
					break;
				case IsoMeasure::FieldLevel::L1:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
					break;
				case IsoMeasure::FieldLevel::L2:
					value_hit0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
					break;
				}
				if (value_hit0 > raycastingOptions.isoValue_0)
				{
					float3 gradient0;

					switch (raycastingOptions.fieldLevel_0)
					{
					case IsoMeasure::FieldLevel::L0:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
						break;
					case IsoMeasure::FieldLevel::L1:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
						break;
					case IsoMeasure::FieldLevel::L2:
						gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
						break;
					}
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					float3 rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparency_0 + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparency_0);
					float depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };

					while (t < NearFar.y && value_hit0 > raycastingOptions.isoValue_0)
					{


						// Position of the isosurface
						float3 position = pixelPos + (rayDir * t);
						position += d_boundingBox.gridDiameter / 2.0;
						//Relative position calculates the position of the point on the CUDA texture

						texPos_L0 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
						texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 2);
						texPos_L2 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 4);



						// it true then we have the second hit (L1)
						switch (raycastingOptions.fieldLevel_1)
						{
						case IsoMeasure::FieldLevel::L0:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
							break;
						case IsoMeasure::FieldLevel::L1:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
							break;
						case IsoMeasure::FieldLevel::L2:
							value_hit1 = callerValueAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
							break;
						}



						if (value_hit1 > raycastingOptions.isoValue_1)
						{

							float3 gradient1;

							switch (raycastingOptions.fieldLevel_1)
							{
							case IsoMeasure::FieldLevel::L0:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L0, d_boundingBox.gridDiameter, gridSize_L0);
								break;
							case IsoMeasure::FieldLevel::L1:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field1, texPos_L1, d_boundingBox.gridDiameter, gridSize_L1);
								break;
							case IsoMeasure::FieldLevel::L2:
								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_0, field2, texPos_L2, d_boundingBox.gridDiameter, gridSize_L2);
								break;
							}


							float angle = dot(gradient1, rayDir);
							if (angle < 0)
							{

								break;
							}
							else
							{
								float3 rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								float depth1 = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
								rgb = rgb1 * raycastingOptions.transparency_0 + (1 - raycastingOptions.transparency_0)*rgb;
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


__device__ float callerValueAtTexLOD
(
	int isoMeasure,
	cudaTextureObject_t & field_0, cudaTextureObject_t & field_1, cudaTextureObject_t & field_2,
	float3 & texPos_L0, float3 & texPos_L1, float3 & texPos_L2,
	float3 & gridDiameter,
	int3 & gridSize_L0, int3 & gridSize_L1, int3 & gridSize_L2,
	int level
)
{
	float value;
	switch (level)
	{
	case 0:
		value =  callerValueAtTex(isoMeasure, field_0, texPos_L0, gridDiameter, gridSize_L0);
		break;
	case 1:
		value =  callerValueAtTex(isoMeasure, field_1, texPos_L1, gridDiameter, gridSize_L1);
		break;
	case 2:
		value =  callerValueAtTex(isoMeasure, field_2, texPos_L2, gridDiameter, gridSize_L2);
		break;
	}
	return value;
}

__device__ float3 callerGradientAtTexLOD
(
	int &isoMeasure,
	cudaTextureObject_t & field_0, cudaTextureObject_t & field_1, cudaTextureObject_t & field_2,
	float3 & texPos_L0, float3 & texPos_L1, float3 & texPos_L2,
	float3 & gridDiameter,
	int3 & gridSize_L0, int3 & gridSize_L1, int3 & gridSize_L2,
	int & level
)
{
	float3 value;
	switch (level)
	{
	case 0:
		value = callerGradientAtTex(isoMeasure, field_0, texPos_L0, gridDiameter, gridSize_L0);
		break;
	case 1:
		value = callerGradientAtTex(isoMeasure, field_1, texPos_L1, gridDiameter, gridSize_L1);
		break;
	case 2:
		value = callerGradientAtTex(isoMeasure, field_2, texPos_L2, gridDiameter, gridSize_L2);
		break;
	}

	return value;
}


__global__ void CudaIsoSurfacRenderer_Multiscale_Defect
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;


			int3 gridSize_L0 = d_boundingBox.gridSize;
			int3 gridSize_L1 = gridSize_L0 / 2;
			int3 gridSize_L2 = gridSize_L1 / 2;

			float value_hit0;
			float value_hit1;

			float3 position_in;
			float3 position_out;
			float3 texPos_out;

			float3 gradient0;
			float3 gradient1;

			float3 rgb1;
			float depth1;

			float3 rgb;
			float depth;
			
			bool hitOutside = false;

			while (t < NearFar.y)
			{
				position_in = pixelPos + (rayDir * t);
				position_in += d_boundingBox.gridDiameter / 2.0;
				
				float3 texPos_L0 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L0);
				float3 texPos_L1 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L1);
				float3 texPos_L2 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L2);

				value_hit0 = callerValueAtTexLOD(raycastingOptions.isoMeasure_0,
					field0, field1, field2,
					texPos_L0, texPos_L1, texPos_L2,
					d_boundingBox.gridDiameter,
					gridSize_L0, gridSize_L1, gridSize_L2,
					raycastingOptions.fieldLevel_0);
				value_hit1 = callerValueAtTexLOD(raycastingOptions.isoMeasure_0,
					field0, field1, field2,
					texPos_L0, texPos_L1, texPos_L2,
					d_boundingBox.gridDiameter,
					gridSize_L0, gridSize_L1, gridSize_L2,
					raycastingOptions.fieldLevel_1
				);

				if (value_hit1 > raycastingOptions.isoValue_1 && hitOutside == false && value_hit0 < raycastingOptions.isoValue_0)
				{

					hitOutside = true; // hit with the second field before the first 
					position_out = position_in;

					if (raycastingOptions.binarySearch)
					{
						float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;

						switch (raycastingOptions.fieldLevel_1)
						{
						case IsoMeasure::FieldLevel::L0:
							position_out = binarySearch(field0, position_in, d_boundingBox.gridDiameter, gridSize_L0, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_out = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize_L0);

							break;
						case IsoMeasure::FieldLevel::L1:
							position_out = binarySearch(field1, position_in, d_boundingBox.gridDiameter, gridSize_L1, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_out = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize_L1);
							break;
						case IsoMeasure::FieldLevel::L2:
							position_out = binarySearch(field2, position_in, d_boundingBox.gridDiameter, gridSize_L2, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_out = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize_L2);
							break;
						}

					}
					else
					{
						switch (raycastingOptions.fieldLevel_1)
						{
						case IsoMeasure::FieldLevel::L0:
							texPos_out = texPos_L0;
							break;
						case IsoMeasure::FieldLevel::L1:
							texPos_out = texPos_L1;
							break;
						case IsoMeasure::FieldLevel::L2:
							texPos_out = texPos_L2;
							break;
						}
					}


					
					
				}

				if (value_hit0 > raycastingOptions.isoValue_0)
				{


					if (raycastingOptions.binarySearch)
					{
						float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;

						switch (raycastingOptions.fieldLevel_0)
						{
						case IsoMeasure::FieldLevel::L0:
							position_in = binarySearch(field0, position_in, d_boundingBox.gridDiameter, gridSize_L0, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_L0 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L0);

							break;
						case IsoMeasure::FieldLevel::L1:
							position_in = binarySearch(field1, position_in, d_boundingBox.gridDiameter, gridSize_L1, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_L1 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L1);
							break;
						case IsoMeasure::FieldLevel::L2:
							position_in = binarySearch(field2, position_in, d_boundingBox.gridDiameter, gridSize_L2, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos_L2 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L2);
							break;
						}

					}
					gradient0 = callerGradientAtTexLOD(raycastingOptions.isoMeasure_0,
						field0, field1, field2,
						texPos_L0, texPos_L1, texPos_L2,
						d_boundingBox.gridDiameter,
						gridSize_L0, gridSize_L1, gridSize_L2,
						raycastingOptions.fieldLevel_0);

					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					rgb = { 1,1,1 };
					rgb = rgb * raycastingOptions.transparency_0 + Array2Float3(raycastingOptions.color_0) * diffuse*(1 - raycastingOptions.transparency_0);
					depth = depthfinder(position_in, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };



					if (magnitude(position_in - position_out) < raycastingOptions.tolerance_0)
					{
						hitOutside = false;
					}


					switch (hitOutside)
					{
					case true:


						gradient1 = callerGradientAtTexLOD(raycastingOptions.isoMeasure_0,
							field0, field1, field2,
							texPos_out, texPos_out, texPos_out,
							d_boundingBox.gridDiameter,
							gridSize_L0, gridSize_L1, gridSize_L2,
							raycastingOptions.fieldLevel_1);

						rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						depth1 = depthfinder(position_out, eyePos, d_boundingBox.m_viewDir, f, n);

						//rgb = rgb1 * raycastingOptions.transparecny + (1 - raycastingOptions.transparecny)*rgb;
						rgb = rgb1;
						rgba = { rgb.x, rgb.y, rgb.z, depth1 };

						break;

					case false:
						while (t < NearFar.y && value_hit0 > raycastingOptions.isoValue_0)
						{
							// Position of the isosurface
							float3 position = pixelPos + (rayDir * t);
							position += d_boundingBox.gridDiameter / 2.0;
							//Relative position calculates the position of the point on the CUDA texture

							texPos_L0 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);
							texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 2);
							texPos_L2 = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize / 4);


							// it true then we have the second hit (L1)
							value_hit1 = callerValueAtTexLOD(raycastingOptions.isoMeasure_0,
								field0, field1, field2,
								texPos_L0, texPos_L1, texPos_L2,
								d_boundingBox.gridDiameter,
								gridSize_L0, gridSize_L1, gridSize_L2,
								raycastingOptions.fieldLevel_1);

							if (value_hit1 > raycastingOptions.isoValue_1) 
							{

								if (raycastingOptions.binarySearch)
								{
									float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;

									switch (raycastingOptions.fieldLevel_1)
									{
									case IsoMeasure::FieldLevel::L0:
										position = binarySearch(field0, position, d_boundingBox.gridDiameter, gridSize_L0, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
										texPos_L0 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L0);

										break;
									case IsoMeasure::FieldLevel::L1:
										position = binarySearch(field1, position, d_boundingBox.gridDiameter, gridSize_L1, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
										texPos_L1 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L1);
										break;
									case IsoMeasure::FieldLevel::L2:
										position = binarySearch(field2, position, d_boundingBox.gridDiameter, gridSize_L2, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
										texPos_L2 = world2Tex(position_in, d_boundingBox.gridDiameter, gridSize_L2);
										break;
									}

								}


								gradient1 = callerGradientAtTexLOD(raycastingOptions.isoMeasure_0,
									field0, field1, field2,
									texPos_L0, texPos_L1, texPos_L2,
									d_boundingBox.gridDiameter,
									gridSize_L0, gridSize_L1, gridSize_L2,
									raycastingOptions.fieldLevel_1);

								float angle = dot(gradient1, rayDir);
								if (angle < 0)
								{

									break;
								}
								else
								{
									rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
									depth1 = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
									rgb = rgb1 * raycastingOptions.transparency_0 + (1 - raycastingOptions.transparency_0)*rgb;
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
					}

					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}
				if (raycastingOptions.adaptiveSampling)
					t += findExitPoint3D(position_in, dir, cellSize);
				else
					t = t + raycastingOptions.samplingRate_0;
				
			}

		}

	}


}


__global__ void CudaIsoSurfacRenderer_Double_Advanced
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t = NearFar.x;
	
			float alpha_primary = 1 - raycastingOptions.transparency_0;
			float alpha_secondary = 1 - raycastingOptions.transparency_1; 
			float alpha_f = 0;


			int3 gridSize = d_boundingBox.gridSize;
			float3 gridDiameter = d_boundingBox.gridDiameter;

			float value_0;
			float value_1;

			float3 position;
			float3 position_out;
			float3 texPos_out;

			float3 gradient0;
			float3 gradient1;

			float3 rgb1;
			float depth1;

			float3 rgb;
			float depth;
			
			bool firstHitOutsideSecondary = false;

			while (t < NearFar.y)
			{
				position = pixelPos + (rayDir * t);
				position += d_boundingBox.gridDiameter / 2.0;
				float3 texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);


				value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
				value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);

				if (value_1 > raycastingOptions.isoValue_1 && firstHitOutsideSecondary == false && value_0 < raycastingOptions.isoValue_0)
				{

					firstHitOutsideSecondary = true;
					position_out = position;
					texPos_out = texPos;

					if (raycastingOptions.binarySearch)
					{
						float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;
						position_out = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos_out = world2Tex(position_out, gridDiameter, gridSize);
					}

				}

				// if the first hit
				if (value_0 > raycastingOptions.isoValue_0)
				{

					if (raycastingOptions.binarySearch)
					{
						float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
						position = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos = world2Tex(position, gridDiameter, gridSize);
					}

					gradient0 = callerGradientAtTex(raycastingOptions.isoMeasure_0,field0, texPos,gridDiameter,gridSize);
					float diffuse = max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					rgb = { 1,1,1 };
					//rgb = Array2Float3(raycastingOptions.color_0) * powf(diffuse)* alpha_primary + rgb * (1-alpha_primary) ; // The primary color blend with with background
					rgb = Array2Float3(raycastingOptions.color_0) * powf(diffuse,2)* alpha_primary + rgb * (1-alpha_primary) ; // The primary color blend with with background
					depth = depthfinder(position, eyePos, d_boundingBox.m_viewDir, f, n);
					float4 rgba = { rgb.x, rgb.y, rgb.z, depth };


					// When they are too close
					if (magnitude(position - position_out) < raycastingOptions.tolerance_0)
					{
						firstHitOutsideSecondary = false;
					}

					switch (firstHitOutsideSecondary)
					{
					case true: // If the first hit with the second isosurface is outside of the primary isosurface

						gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos_out, gridDiameter, gridSize);
						rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						depth1 = depthfinder(position_out, eyePos, d_boundingBox.m_viewDir, f, n);

						rgb = rgb1 * alpha_secondary + rgb * alpha_primary * (1 - alpha_secondary);
						alpha_f = alpha_secondary + (1 - alpha_secondary) * alpha_primary; // final Alpha value
						rgb = rgb * alpha_f + make_float3(1, 1, 1) * (1 - alpha_f);
						
						rgba = { rgb.x, rgb.y, rgb.z, depth1 };

						break;

					case false: // If the first hit with the second isosurface is inside the primary isosurface

						while (t < NearFar.y && value_0 > raycastingOptions.isoValue_0)
						{
							position = pixelPos + (rayDir * t);
							position += d_boundingBox.gridDiameter / 2.0;
							texPos = world2Tex(position, d_boundingBox.gridDiameter, d_boundingBox.gridSize);

							value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);

							if (value_1 > raycastingOptions.isoValue_1)
							{

								if (raycastingOptions.binarySearch)
								{
										float3 samplingVector = rayDir * raycastingOptions.samplingRate_1;
										position = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
										texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
								}

								gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);
								float angle = dot(gradient1, rayDir);

								if (angle > 0)
								{
									rgb1 = (Array2Float3(raycastingOptions.color_1) ^ raycastingOptions.brightness) * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
									rgb = rgb * alpha_primary + rgb1 * alpha_secondary * (1 - alpha_primary);
									alpha_f = alpha_primary + (1 - alpha_primary) * alpha_secondary; // final Alpha value
									rgb = rgb * alpha_f + make_float3(1, 1, 1) * (1 - alpha_f);

									rgba = { rgb.x, rgb.y, rgb.z, depth };
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
						break;
					}

					surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
					break;

				}

				if (raycastingOptions.adaptiveSampling)
					t += findExitPoint3D(position, dir, cellSize);
				else
					t = t + raycastingOptions.samplingRate_0;

			}

		}

	}


}




//__global__ void CudaIsoSurfacRenderer_Double_Transparency
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t field0,
//	cudaTextureObject_t field1,
//	int rays,
//	RaycastingOptions raycastingOptions,
//	RenderingOptions renderingOptions
//)
//{
//	int index = CUDA_INDEX;
//
//	if (index < rays)
//	{
//		// determine pixel position based on the index of the thread
//		int2 pixel = index2pixel(index, d_boundingBox.m_width);
//		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
//		int3 gridInterval = make_int3(d_boundingBox.gridSize.x - 1, d_boundingBox.gridSize.y - 1, d_boundingBox.gridSize.z - 1);
//		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
//		float3 dir = normalize(pixelPos - d_boundingBox.m_eyePos);
//		float2 NearFar = findEnterExit(pixelPos, dir, d_boundingBox.boxFaces);
//
//
//		// if inside the bounding box
//		if (NearFar.y != -1)
//		{
//
//			float3 rayDir = normalize(pixelPos - d_boundingBox.m_eyePos);
//			// near and far plane
//			float n = renderingOptions.nearField;
//			float f = renderingOptions.farField;
//			// Add the offset to the eye position
//			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
//			float t_0 = NearFar.x;
//			float t_1 = NearFar.x;
//
//
//
//			int3 gridSize = d_boundingBox.gridSize;
//			float3 gridDiameter = d_boundingBox.gridDiameter;
//			float3 view = normalize(d_boundingBox.m_viewDir);
//
//			float value_0;
//			float value_1;
//
//			
//			float3 position_out;
//			float3 texPos_out;
//
//			float3 gradient0;
//			float3 gradient1;
//
//			float3 rgb0 = Array2Float3(raycastingOptions.color_0);
//			float3 rgb1 = Array2Float3(raycastingOptions.color_1);
//			float3 lightColor = Array2Float3(renderingOptions.lightColor);
//
//
//			float3 rgb = rgb0;
//			float4 rgba;
//			float depth;
//
//			float enter_0[4] = { 0,0,0,0};
//			float enter_R_0[4] = { 0,0,0,0 }; //Reflection
//
//			//
//			//float exit_0[4] = { NearFar.y,NearFar.y,NearFar.y,NearFar.y };
//			//float exit_R_0[4] = { 0,0,0,0 }; //Reflection
//
//			//float enter_1[4] = { 0,0,0,0 };
//			//float exit_1[4] = { 0,0,0,0 };
//
//			int hitLimit = 10;
//
//			int nHit_0 = 0;
//			int nHit_1 = 0;
//
//			float Reflection;
//			float Alpha = 0;
//
//			float3 texPos;
//			float3 initialPos = pixelPos + d_boundingBox.gridDiameter / 2.0;
//			float3 position = initialPos;
//			float3 firstHit;
//
//			float3 bgColor = Array2Float3(renderingOptions.bgColor);
//
//			bool insideStructure = false;
//		
//
//			while (t_0 < NearFar.y )
//			{
//				position = initialPos + (rayDir * t_0);
//				texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
//				value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
//				value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);
//
//
//				// Check for outside hits of the secondary isosurface
//				if (value_1 > raycastingOptions.isoValue_1 && !raycastingOptions.insideOnly)
//				{
//					nHit_0++;
//					if (raycastingOptions.binarySearch)
//					{
//						float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
//						position_out = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
//						texPos = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize);
//					}
//		
//					gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);
//					rgb = rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f) + (1, 1, 1) * powf(max(dot(normalize(gradient1), view),0.0f),raycastingOptions.shininess) * raycastingOptions.specularCoefficient;
//
//					firstHit = position;
//					break;
//
//					
//				}
//				
//				// inside the primary 
//				if (value_0 > raycastingOptions.isoValue_0)
//				{
//
//					if (raycastingOptions.binarySearch)
//					{
//						float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
//						position_out = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
//						texPos = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize);
//
//					}
//
//					if (nHit_0 == 0)
//						firstHit = position;
//
//					gradient0 = normalize(callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize));
//					Reflection = 1 - max(powf(fabsf(dot(gradient0, view)), raycastingOptions.reflectionCoefficient), 0.2f);
//					Alpha = Reflection + (1 - Reflection)*Alpha;
//					rgb = (1 - Alpha) * bgColor;
//					
//
//					
//					// Check for the second isosurface
//					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y)
//					{
//						nHit_0++;
//
//						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
//						value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);
//
//						
//						if (value_1 > raycastingOptions.isoValue_1)
//						{
//							if (raycastingOptions.binarySearch)
//							{
//								float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
//								position_out = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
//								texPos = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize);
//							}
//
//							gradient1 = callerGradientAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);
//							//float angle = dot(gradient1, rayDir);
//
//							//if (angle < 0)
//							//	break;
//							//else
//							//{
//								rgb1 = rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f) + (1, 1, 1) * powf(max(dot(normalize(gradient1), view), 0.0f), raycastingOptions.shininess) * raycastingOptions.specularCoefficient;
//								rgb = (1 - Alpha) * rgb1 + Alpha * rgb;
//								insideStructure = true;
//								break;
//							//}
//						}
//	
//						if (raycastingOptions.adaptiveSampling)
//						{
//							t_0 += findExitPoint3D(position, dir, cellSize);;
//						}
//						else
//						{
//							t_0 += raycastingOptions.samplingRate_0;
//						}
//
//						position = initialPos + (rayDir * t_0);
//						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
//
//					}
//
//					if (insideStructure)
//						break;
//
//					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y)
//					{
//
//						if (raycastingOptions.adaptiveSampling)
//						{
//							t_0 += findExitPoint3D(position, dir, cellSize);;
//						}
//						else
//						{
//							t_0 += raycastingOptions.samplingRate_0;
//						}
//
//						position = initialPos + (rayDir * t_0);
//						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
//						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
//
//					}
//
//					if (t_0 < NearFar.y)
//					{
//
//						if (raycastingOptions.binarySearch)
//						{
//							float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;
//							position_out = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
//							texPos = world2Tex(position_out, d_boundingBox.gridDiameter, gridSize);
//
//						}
//
//						gradient0 = normalize(callerGradientAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize));
//						Reflection = 1 - max(powf(fabsf(dot(gradient0, view)), raycastingOptions.reflectionCoefficient), 0.2f);
//						Alpha = Reflection + (1 - Reflection)*Alpha;
//						rgb = (1 - Alpha) *bgColor;
//						
//
//					}
//
//
//				}
//
//				if (raycastingOptions.adaptiveSampling)
//				{
//					t_0 += findExitPoint3D(position, dir, cellSize);;
//				}
//				else
//				{
//					t_0 += raycastingOptions.samplingRate_0;
//				}
//
//			} 
//
//			if (nHit_0 != 0)
//			{
//
//				
//				depth = depthfinder(firstHit , eyePos, d_boundingBox.m_viewDir, f, n);
//
//				rgba = { rgb.x, rgb.y, rgb.z, depth };
//				surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);
//
//			}
//
//		}
//	}
//}



__global__ void CudaIsoSurfacRenderer_Double_Transparency_noglass
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t_0 = NearFar.x;




			int3 gridSize = d_boundingBox.gridSize;
			float3 gridDiameter = d_boundingBox.gridDiameter;
			float3 view = normalize(d_boundingBox.m_viewDir);

			float value_0;
			float value_1;

			float3 gradient0;
			float3 gradient1;

			float3 rgb0 = Array2Float3(raycastingOptions.color_0);
			float3 rgb1 = Array2Float3(raycastingOptions.color_1);

			float depth;

			float3 bgColor = Array2Float3(renderingOptions.bgColor);
			float3 lightColor = Array2Float3(renderingOptions.lightColor);
			float3 rgb = bgColor;

			float4 rgba;

			int nHit = 0;
			float alpha_trans = raycastingOptions.transparency_1;
			float Alpha = 0;

			float3 diffuse;
			float3 specular;

			float3 texPos;
			float3 initialPos = pixelPos + d_boundingBox.gridDiameter / 2.0;
			float3 position = initialPos;
			float3 firstHitPosition;
			float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;



			bool insideStructure = false;
			float distanceToNormal = 0;

			while (t_0 < NearFar.y)
			{
				position = initialPos + (rayDir * t_0);
				texPos = world2Tex(position, gridDiameter, gridSize);
				value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
				value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);


				 //Check for outside hits of the secondary isosurface
				if (value_1 > raycastingOptions.isoValue_1 && !raycastingOptions.insideOnly)
				{

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
					}

					if (raycastingOptions.normalCurves)
					{
						distanceToNormal = distanceToNormalCurves(
							raycastingOptions.isoMeasure_0,
							raycastingOptions.isoValue_0,
							position,
							field0,
							gridSize,
							gridDiameter,
							raycastingOptions.samplingRate_1);

						gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);
						rgb1 = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distanceToNormal, renderingOptions.minMeasure, renderingOptions.maxMeasure);
						diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;
						rgb = diffuse + specular;
						if(nHit == 0)
							firstHitPosition = position;
						Alpha = 1;
						nHit++;
						break;
					}
					else
					{
						gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);
						diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;
						rgb = diffuse + specular;
						if (nHit == 0)
							firstHitPosition = position;
						Alpha = 1;
						nHit++;
						break;
					}
				}
				if (raycastingOptions.secondaryOnly)
				{
					if (raycastingOptions.adaptiveSampling)
					{
						t_0 += findExitPoint3D(position, dir, cellSize);;
					}
					else
					{
						t_0 += raycastingOptions.samplingRate_0;
					}
					continue;
				}
				// inside the primary 
				if (value_0 > raycastingOptions.isoValue_0 && Alpha < 1)
				{

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos = world2Tex(position, gridDiameter, gridSize);
					}

					if (nHit == 0)
						firstHitPosition = position;


					gradient0 = gradientTrilinear(raycastingOptions.isoMeasure_0, texPos, field0, gridSize, gridDiameter);
					diffuse = renderingOptions.Kd *rgb0 * max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					specular = lightColor * powf(max(dot(normalize(gradient0), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks;


					rgb = (1 - Alpha) * alpha_trans * (diffuse + specular) + Alpha * rgb;
					Alpha = Alpha + (1 - Alpha)*alpha_trans;

					nHit++;
					

					// Check for the second isosurface
					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y && Alpha < 1)
					{

						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);
						value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);


						if (value_1 > raycastingOptions.isoValue_1)
						{
							if (raycastingOptions.binarySearch)
							{
								position = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
								texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
							}


							gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);
							
							if (dot(normalize(gradient1), d_boundingBox.m_viewDir) > 0) // removes 
							{
								diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;


								rgb = (1 - Alpha) * 1 * (diffuse + specular) + Alpha * rgb;
								Alpha = 1;

								insideStructure = true;
								break;
							}

						}

						if (raycastingOptions.adaptiveSampling)
						{
							t_0 += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t_0 += raycastingOptions.samplingRate_0;
						}

						position = initialPos + (rayDir * t_0);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);

					}

					if (insideStructure)
						break;

					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y && Alpha < 1)
					{

						if (raycastingOptions.adaptiveSampling)
						{
							t_0 += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t_0 += raycastingOptions.samplingRate_0;
						}

						position = initialPos + (rayDir * t_0);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos, gridDiameter, gridSize);

					}

					if (t_0 < NearFar.y && Alpha < 1)
					{

						if (raycastingOptions.binarySearch)
						{
							position = binarySearch(field0, position, gridDiameter, gridSize, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);

						}

						gradient0 = gradientTrilinear(raycastingOptions.isoMeasure_0, texPos, field0, gridSize, gridDiameter);
						float3 diffuse = renderingOptions.Kd * rgb0 * max(fabsf(dot(normalize(gradient0), d_boundingBox.m_viewDir)), 0.0f);
						float3 specular = lightColor * powf(max(dot(normalize(gradient0), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks;
	
						rgb = (1 - Alpha) * alpha_trans * (diffuse + specular) +  Alpha * rgb;
						Alpha = Alpha + (1 - Alpha)*alpha_trans;
					}
				}

				if (raycastingOptions.adaptiveSampling)
				{
					t_0 += findExitPoint3D(position, dir, cellSize);;
				}
				else
				{
					t_0 += raycastingOptions.samplingRate_0;
				}

			}

			if (nHit != 0)
			{
				// blend it with background
				rgb = (1 - Alpha) * bgColor + Alpha * rgb;
				depth = depthfinder(firstHitPosition, eyePos, d_boundingBox.m_viewDir, f, n);
				rgba = { rgb.x, rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);

			}

		}
	}
}

	

__global__ void CudaIsoSurfacRenderer_Double_Transparency_noglass_multiLevel
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
		float3 cellSize = d_boundingBox.gridDiameter / gridInterval;
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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			float t_0 = NearFar.x;




			int3 gridSize = d_boundingBox.gridSize;
			int3 gridSize_L1 = d_boundingBox.gridSize / 2.0f;
			float3 gridDiameter = d_boundingBox.gridDiameter;
			float3 view = normalize(d_boundingBox.m_viewDir);

			float value_0;
			float value_1;

			float3 gradient0;
			float3 gradient1;

			float3 rgb0 = Array2Float3(raycastingOptions.color_0);
			float3 rgb1 = Array2Float3(raycastingOptions.color_1);

			float depth;

			float3 bgColor = Array2Float3(renderingOptions.bgColor);
			float3 lightColor = Array2Float3(renderingOptions.lightColor);
			float3 rgb = bgColor;

			float4 rgba;

			int nHit = 0;
			float alpha_trans = raycastingOptions.transparency_1;
			float Alpha = 0;

			float3 diffuse;
			float3 specular;

			float3 texPos;
			float3 texPos_L1;
			float3 initialPos = pixelPos + d_boundingBox.gridDiameter / 2.0;
			float3 position = initialPos;
			float3 firstHitPosition;
			float3 samplingVector = rayDir * raycastingOptions.samplingRate_0;



			bool insideStructure = false;
			float distanceToNormal = 0;

			while (t_0 < NearFar.y)
			{
				position = initialPos + (rayDir * t_0);
				texPos = world2Tex(position, gridDiameter, gridSize);
				texPos_L1 = world2Tex(position, gridDiameter, gridSize_L1);
				value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L1, gridDiameter, gridSize_L1);
				value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);


				//Check for outside hits of the secondary isosurface
				if (value_1 > raycastingOptions.isoValue_1 && !raycastingOptions.insideOnly)
				{

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
					}

					if (raycastingOptions.normalCurves)
					{
						distanceToNormal = distanceToNormalCurves(
							raycastingOptions.isoMeasure_0,
							raycastingOptions.isoValue_0,
							position,
							field0,
							gridSize,
							gridDiameter,
							raycastingOptions.samplingRate_1);

						gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);
						rgb1 = colorCodeRange(renderingOptions.minColor, renderingOptions.maxColor, distanceToNormal, renderingOptions.minMeasure, renderingOptions.maxMeasure);
						diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;
						rgb = diffuse + specular;
						if (nHit == 0)
							firstHitPosition = position;
						Alpha = 1;
						nHit++;
						break;
					}
					else
					{
						gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);
						diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
						specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;
						rgb = diffuse + specular;
						if (nHit == 0)
							firstHitPosition = position;
						Alpha = 1;
						nHit++;
						break;
					}
				}
				if (raycastingOptions.secondaryOnly)
				{
					if (raycastingOptions.adaptiveSampling)
					{
						t_0 += findExitPoint3D(position, dir, cellSize);;
					}
					else
					{
						t_0 += raycastingOptions.samplingRate_0;
					}
					continue;
				}
				// inside the primary 
				if (value_0 > raycastingOptions.isoValue_0 && Alpha < 1)
				{

					if (raycastingOptions.binarySearch)
					{
						position = binarySearch(field0, position, gridDiameter, gridSize_L1, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
						texPos_L1 = world2Tex(position, gridDiameter, gridSize_L1);
					}

					if (nHit == 0)
						firstHitPosition = position;


					gradient0 = gradientTrilinear(raycastingOptions.isoMeasure_0, texPos_L1, field0, gridSize_L1, gridDiameter);
					diffuse = renderingOptions.Kd *rgb0 * max(dot(normalize(gradient0), d_boundingBox.m_viewDir), 0.0f);
					specular = lightColor * powf(max(dot(normalize(gradient0), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks;


					rgb = (1 - Alpha) * alpha_trans * (diffuse + specular) + Alpha * rgb;
					Alpha = Alpha + (1 - Alpha)*alpha_trans;

					nHit++;


					// Check for the second isosurface
					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y && Alpha < 1)
					{

						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L1, gridDiameter, gridSize_L1);
						value_1 = callerValueAtTex(raycastingOptions.isoMeasure_1, field1, texPos, gridDiameter, gridSize);


						if (value_1 > raycastingOptions.isoValue_1)
						{
							if (raycastingOptions.binarySearch)
							{
								position = binarySearch(field1, position, gridDiameter, gridSize, raycastingOptions.isoValue_1, samplingVector, raycastingOptions.isoMeasure_1, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
								texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
							}


							gradient1 = gradientTrilinear(raycastingOptions.isoMeasure_1, texPos, field1, gridSize, gridDiameter);

							if (dot(normalize(gradient1), d_boundingBox.m_viewDir) > 0) // removes 
							{
								diffuse = renderingOptions.Kd1 * rgb1 * max(dot(normalize(gradient1), d_boundingBox.m_viewDir), 0.0f);
								specular = lightColor * powf(max(dot(normalize(gradient1), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks1;


								rgb = (1 - Alpha) * 1 * (diffuse + specular) + Alpha * rgb;
								Alpha = 1;

								insideStructure = true;
								break;
							}

						}

						if (raycastingOptions.adaptiveSampling)
						{
							t_0 += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t_0 += raycastingOptions.samplingRate_0;
						}

						position = initialPos + (rayDir * t_0);
						texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
						texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L1);

					}

					if (insideStructure)
						break;

					while (value_0 > raycastingOptions.isoValue_0 && t_0 < NearFar.y && Alpha < 1)
					{

						if (raycastingOptions.adaptiveSampling)
						{
							t_0 += findExitPoint3D(position, dir, cellSize);;
						}
						else
						{
							t_0 += raycastingOptions.samplingRate_0;
						}

						position = initialPos + (rayDir * t_0);
						texPos_L1 = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
						value_0 = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, texPos_L1, gridDiameter, gridSize_L1);

					}

					if (t_0 < NearFar.y && Alpha < 1)
					{

						if (raycastingOptions.binarySearch)
						{
							position = binarySearch(field0, position, gridDiameter, gridSize_L1, raycastingOptions.isoValue_0, samplingVector, raycastingOptions.isoMeasure_0, raycastingOptions.tolerance_0, raycastingOptions.maxIteration);
							texPos = world2Tex(position, d_boundingBox.gridDiameter, gridSize_L1);

						}

						gradient0 = gradientTrilinear(raycastingOptions.isoMeasure_0, texPos_L1, field0, gridSize, gridDiameter);
						float3 diffuse = renderingOptions.Kd * rgb0 * max(fabsf(dot(normalize(gradient0), d_boundingBox.m_viewDir)), 0.0f);
						float3 specular = lightColor * powf(max(dot(normalize(gradient0), view), 0.0f), renderingOptions.shininess) * renderingOptions.Ks;

						rgb = (1 - Alpha) * alpha_trans * (diffuse + specular) + Alpha * rgb;
						Alpha = Alpha + (1 - Alpha)*alpha_trans;
					}
				}

				if (raycastingOptions.adaptiveSampling)
				{
					t_0 += findExitPoint3D(position, dir, cellSize);;
				}
				else
				{
					t_0 += raycastingOptions.samplingRate_0;
				}

			}

			if (nHit != 0)
			{
				// blend it with background
				rgb = (1 - Alpha) * bgColor + Alpha * rgb;
				depth = depthfinder(firstHitPosition, eyePos, d_boundingBox.m_viewDir, f, n);
				rgba = { rgb.x, rgb.y, rgb.z, depth };
				surf2Dwrite(rgba, raycastingSurface, 4 * sizeof(float) * pixel.x, pixel.y);

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
			float3 eyePos = d_boundingBox.m_eyePos + d_boundingBox.gridDiameter / 2.0;
			int3 gridSize = d_boundingBox.gridSize;


			for (float t = NearFar.x; t < NearFar.y; t = t + raycastingOptions.samplingRate_0)
			{


				// Position of the isosurface
				float3 position = pixelPos + (rayDir * t);

				// Adds an offset to position while the center of the grid is at gridDiamter/2
				position += d_boundingBox.gridDiameter / 2.0;

				bool hit = false;

				switch (raycastingOptions.projectionPlane)
				{

				case(IsoMeasure::ProjectionPlane::ZXPLANE):

					if (position.y < d_boundingBox.gridDiameter.y * raycastingOptions.planeProbePosition &&
						position.y > d_boundingBox.gridDiameter.y * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;
					break;
				case(IsoMeasure::ProjectionPlane::XYPLANE):

					if (position.z < d_boundingBox.gridDiameter.z * raycastingOptions.planeProbePosition &&
						position.z > d_boundingBox.gridDiameter.z * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;
					break;
				case(IsoMeasure::ProjectionPlane::YZPLANE):

					if (position.x < d_boundingBox.gridDiameter.x * raycastingOptions.planeProbePosition &&
						position.x > d_boundingBox.gridDiameter.x * raycastingOptions.planeProbePosition - raycastingOptions.planeThinkness)
						hit = true;

					break;
				}

				if (hit == true)
				{
					float3 relativePos = world2Tex(position, d_boundingBox.gridDiameter, gridSize);
					float value = callerValueAtTex(raycastingOptions.isoMeasure_0, field0, relativePos, d_boundingBox.gridDiameter, gridSize);
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


__device__ float3 gradientTrilinear(int isoMeasure, float3 texPos, cudaTextureObject_t field0, int3 & gridSize, float3 & gridDiameter)
{

	float3 values[8] = 
	{
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5f		, floor(texPos.y - .5) + .5		, floor(texPos.z - .5) + .5f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5 +	1.0f, floor(texPos.y - .5) + .5		, floor(texPos.z - .5) + .5f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5		, floor(texPos.y - .5) + .5 + 1.0f , floor(texPos.z - .5) + .5f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5 + 1.0f, floor(texPos.y - .5) + .5 + 1.0f , floor(texPos.z - .5) + .5f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5		, floor(texPos.y - .5) + .5		, floor(texPos.z - .5) + .5 + 1.0f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5 + 1.0f, floor(texPos.y - .5) + .5		, floor(texPos.z - .5) + .5 + 1.0f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5		, floor(texPos.y - .5) + .5 + 1.0f , floor(texPos.z - .5) + .5 + 1.0f), gridDiameter, gridSize),
		callerGradientAtTex(isoMeasure, field0, make_float3(floor(texPos.x - .5) + .5 + 1.0f, floor(texPos.y - .5) + .5 + 1.0f , floor(texPos.z - .5) + .5 + 1.0f), gridDiameter, gridSize)
	};

	return trilinearInterpolation<float3>(values, texPos);

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



__device__ float distanceToNormalCurves(
	int & isoMeasure,
	float & isoValue,
	float3 position,
	cudaTextureObject_t field0,
	int3 gridSize,
	float3 gridDiameter,
	float samplingStep
) {

	float3 tempPos = position;
	float3 texPos = world2Tex(tempPos, gridDiameter, gridSize);
	float3 dir = normalize(gradientTrilinear(isoMeasure, texPos, field0, gridSize, gridDiameter));
	float value = 0;
	float distance = 0;

	int iteration = 0;

	while ( iteration < 1000 ) {

		tempPos = tempPos + dir * samplingStep;
		texPos = world2Tex(tempPos, gridDiameter, gridSize);
		value = callerValueAtTex(isoMeasure, field0, texPos, gridDiameter, gridSize);
		distance = magnitude(position - tempPos);
		iteration++;

		if (value > isoValue)
		{
			break;
		}
		dir = normalize(gradientTrilinear(isoMeasure, texPos, field0, gridSize, gridDiameter));

	}

	return iteration * samplingStep;

}