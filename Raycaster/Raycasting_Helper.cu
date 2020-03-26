

#include "Raycasting_Helper.h"
#include "../Cuda/helper_math.h"

typedef unsigned char uchar;



// return the far and near intersection with bounding box
__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox)
{

	bool hit = true;


	float arrayPixelPos[3] = { pixelPos.x, pixelPos.y, pixelPos.z };

	float tNear = -10000000;
	float tFar = +10000000;

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