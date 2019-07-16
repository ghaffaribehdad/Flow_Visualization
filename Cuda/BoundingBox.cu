#include "BoundingBox.cuh"
#include <iostream>
#include <fstream>
__device__ __host__ void BoundingBox::constructEyeCoordinates()
{
	// N vector
	this->nuv[0] = normalize(this->viewDir);

	// U vector
	float3 upVectNorm = normalize(this->upVec);
	this->nuv[1] = cross(nuv[0], upVectNorm);

	// V vector
	this->nuv[2] = cross(nuv[0], nuv[1]);
}


__device__ float3 BoundingBox::pixelPosition(const int& i, const int& j)
{
	// Height of the Image Plane
	float H = static_cast<float>(tan(this->FOV / 2.0) * 2.0 * this->distImagePlane);
	// Width of the Image Plane
	float W = H * this->aspectRatio;

	// Center of Image Plane
	float3 centerPos = (this->eyePos + this->nuv[0]) * this->distImagePlane;

	// Left Corner of Image Plane
	float3 leftCornerPos = (centerPos + (this->nuv[1] * W / 2.0f) - (this->nuv[2] * H / 2.0f));

	float3 pixelPos = leftCornerPos - (this->nuv[1] * i * W / this->width);
	pixelPos += this->nuv[2] * j * H / this->height;

	return pixelPos;
}



__device__ float2 BoundingBox::findIntersections(float3& pixelPos)
{

	bool hit = true;


	float arrayEyePos[3] = { this->eyePos.x,this->eyePos.y,this->eyePos.z };
	float arrayPixelPos[3] = { pixelPos.x, pixelPos.y, pixelPos.z };

	float tNear = -1000;
	float tFar = +1000;

	// iterates over x,y,z planes
	for (int i = 0; i < 3; i++)
	{
		float plane1 = this->boxFaces[2 * i];
		float plane2 = this->boxFaces[2 * i + 1];
		float t1 = 0;
		float t2 = 0;


		float D = arrayPixelPos[i] - arrayEyePos[i];

		// check if ray and axis are aligned
		if (D == 0)
		{
			if (arrayPixelPos[i] < plane1 || arrayPixelPos[i] > plane2)
			{
				hit = false;
				break;
			}
		}
		else
		{
			t1 = (plane1 - arrayPixelPos[i]) / D;
			float tTemp = (plane2 - arrayPixelPos[i]) / D;

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

//void BoundingBox::reconstruct()
//{
//	std::ofstream myfile;
//	myfile.open("example.csv");
//	
//
//	
//	constructEyeCoordinates();
//	for (int i = 0; i < width; i++)
//	{
//		for (int j = 0; j < height; j++)
//		{
//			Ray ray;
//			ray.setPixel(i, j);
//			ray.setPixelPos = pixelPosition(i, j);
//			
//			float2 t = findIntersections(ray.getPixelPos());
//			
//			if (t.x == -1 )
//			{
//				continue;
//			}
//			else
//			{
//				
//				float3 position = ray.getPos(t.x);
//				//std::cout <<"[" << position.x << ", " << position.y << ", " << position.z << ", " << "]\n";
//				myfile << position.x << ", " << position.y << "," << position.z << "," << "\n";
//				position = ray.getPos(t.y);
//				//std::cout <<"[" << position.x << ", " << position.y << ", " << position.z << ", " << "]\n";
//				myfile << position.x << ", " << position.y << "," << position.z << "," << "\n";
//			}
//			
//		}
//	}
//	myfile.close();
//}


__device__ __host__ void BoundingBox::initialize()
{
	this->updateAspectRatio();
	this->updateBoxFaces();
	this->constructEyeCoordinates();
}

__host__ __device__ void BoundingBox::updateBoxFaces()
{
	for (int i = 0; i < 3; i++)
	{
		this->boxFaces[2 * i] = this->gridDiameter.x / -2.0f;
		this->boxFaces[2 * i + 1] = this->gridDiameter.x / 2.0f;
	}
}

__host__ __device__ void BoundingBox::updateAspectRatio()
{
	this->aspectRatio = this->width / this->height;
}