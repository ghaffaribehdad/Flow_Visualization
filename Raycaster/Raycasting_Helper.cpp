

#include "Raycasting_Helper.h"

typedef unsigned char uchar;




__device__ float2 findIntersections(const float3 pixelPos, const BoundingBox boundingBox)
{

	bool hit = true;


	float arrayPixelPos[3] = { pixelPos.x, pixelPos.y, pixelPos.z };

	float tNear = -1000;
	float tFar = +1000;

	float3 _D = normalize(pixelPos - boundingBox.eyePos);
	float D[3] = { _D.x,_D.y,_D.z };

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
			//

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
		return { tNear,tFar };
	}
	else
	{
		return { -1,-1 };
	}

}


__device__ float3 pixelPosition(const BoundingBox  boundingBox, const int i, const int j)
{
	// Height of the Image Plane
	float H = static_cast<float>(tan(boundingBox.FOV / 2.0) * 2.0 * boundingBox.distImagePlane);
	// Width of the Image Plane
	float W = H * boundingBox.aspectRatio;

	// Center of Image Plane
	float3 centerPos = (boundingBox.eyePos + boundingBox.nuv[0]) * boundingBox.distImagePlane;

	// Left Corner of Image Plane
	float3 leftCornerPos = (centerPos + (boundingBox.nuv[1] * W / 2.0f) - (boundingBox.nuv[2] * H / 2.0f));

	float3 pixelPos = leftCornerPos - (boundingBox.nuv[1] * float(i) * W / float(boundingBox.width));
	pixelPos += boundingBox.nuv[2] * float(j) * H / float(boundingBox.height);

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