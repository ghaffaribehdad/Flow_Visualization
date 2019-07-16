#include "BoundingBox.cuh"

__host__ __device__ void BoundingBox::constructEyeCoordinates()
{
	// N vector
	this->nuv[0] = normalize(this->viewDir);
	
	// U vector
	float3 upVectNorm = normalize(this->upVec);
	this->nuv[1]= cross(upVectNorm, nuv[0]);

	// V vector
	this->nuv[2] = cross(nuv[0], nuv[1]);
}


__host__ __device__ float3 BoundingBox::pixelPosition(const int& i, const int& j)
{
	// Height of the Image Plane
	float H = static_cast<float>(tan(this->FOV / 2.0) * 2.0 * this->distImagePlane);
	// Width of the Image Plane
	float W = H * this->aspectRatio;

	// Center of Image Plane
	float3 centerPos = (this->eyePos - this->nuv[0]) * this->distImagePlane;

	// Left Corner of Image Plane
	float3 leftCornerPos = (centerPos - this->nuv[1]) * (W / 2.0f) - (this->nuv[2] * H / 2.0f);

	float3 pixelPos = leftCornerPos + (this->nuv[1] * i * W / this->width);
	pixelPos += this->nuv[2] * j * H / this->height;

	return pixelPos;
}

__host__ __device__  Ray BoundingBox::createRayfromPixel(int i, int j)
{
	float3 pixelPos = pixelPosition(i, j);

	Ray ray = Ray(this->eyePos, pixelPos);

	return ray;
}

__host__ __device__ float BoundingBox::findIntersections(Ray & ray)
{
	bool hit = true;
	float tNear = -1000;
	float tFar = +1000;

	float arrayEyePos[3] = { this->eyePos.x,this->eyePos.y,this->eyePos.z };
	float arrayPixelPos[3] = { ray.getPixelPos().x, ray.getPixelPos().y, ray.getPixelPos().z };

	while (hit)
	{
		// iterates over x,y,z planes
		for (int i = 0; i < 3; i++)
		{
			float plane1 = this->boxFaces[i];
			float plane2 = this->boxFaces[i + 1];
			float t1 = 0;
			float t2 = 0;


			float D = arrayEyePos[0] - arrayPixelPos[0];

			if (D == 0)
			{
				if (arrayPixelPos[0] < plane1 || arrayPixelPos[0] > plane2)
				{
					hit = false;
					break;
				}
			}
			else
			{
				t1 = (plane1 - arrayPixelPos[0]) / D;
				float tTemp = (plane2 - arrayPixelPos[1]) / D;
				
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
	}
	
	if (hit)
	{
		return tNear;
	}

	else
	{
		return -1;
	}
}