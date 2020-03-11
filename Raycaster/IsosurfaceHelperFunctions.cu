
#include "IsosurfaceHelperFunctions.h"
#include "..//Cuda/helper_math.h"
#include "BoundingBox.h"
#include "..//Cuda/CudaHelperFunctions.h"




__device__ float3 IsosurfaceHelper::Observable::GradientAtXYZ(cudaTextureObject_t tex, float3 relativePos, int3 gridSize)
{
	int3 gridPoint_0 = floorMult(relativePos, gridSize);
	int3 gridPoint_1 = ceilMult(relativePos, gridSize);


	return { 0,0,0 };
}

__device__  float3 IsosurfaceHelper::Observable::GradientAtGrid(cudaTextureObject_t tex, float3 position, int3 gridSize)
{
	float3 h = { 1.0f, 1.0f ,1.0f };
	h =  h/gridSize;
	float dV_dX = this->ValueAtXYZ(tex, make_float3(position.x + 1 , position.y, position.z));
	float dV_dY = this->ValueAtXYZ(tex, make_float3(position.x, position.y + 1 , position.z));
	float dV_dZ = this->ValueAtXYZ(tex, make_float3(position.x, position.y, position.z + 1));

	dV_dX -= this->ValueAtXYZ(tex, make_float3(position.x - 1 , position.y, position.z));
	dV_dY -= this->ValueAtXYZ(tex, make_float3(position.x, position.y - 1, position.z));
	dV_dZ -= this->ValueAtXYZ(tex, make_float3(position.x, position.y, position.z - 1 ));

	return { dV_dX / (2.0f * h.x) ,dV_dY / (2.0f * h.y), dV_dZ / (2.0f * h.z) };
}


__device__  float3 GradientAtGrid_X(cudaTextureObject_t tex, float3 position, int3 gridSize)
{
	float3 h = { 1.0f, 1.0f ,1.0f };
	h = h / gridSize;
	float dV_dX = ValueAtXYZ_float4(tex, make_float3(position.x + 1, position.y, position.z)).x;
	float dV_dY = ValueAtXYZ_float4(tex, make_float3(position.x, position.y + 1, position.z)).x;
	float dV_dZ = ValueAtXYZ_float4(tex, make_float3(position.x, position.y, position.z + 1)).x;

	dV_dX -= ValueAtXYZ_float4(tex, make_float3(position.x - 1, position.y, position.z)).x;
	dV_dY -= ValueAtXYZ_float4(tex, make_float3(position.x, position.y - 1, position.z)).x;
	dV_dZ -= ValueAtXYZ_float4(tex, make_float3(position.x, position.y, position.z - 1)).x;

	return { dV_dX / h.x ,dV_dY / h.y, dV_dZ / h.z };
}


__device__ float IsosurfaceHelper::Velocity_Magnitude::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 _velocity = tex3D<float4>(tex, position.x, position.y, position.z);
	float3 velocity = make_float3(_velocity.x, _velocity.y, _velocity.z);
	return fabsf(sqrtf(dot(velocity, velocity)));
}

__device__ float4 IsosurfaceHelper::Velocity_XYZT::ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z);
}

__device__ float IsosurfaceHelper::Velocity_X::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z).x;
}

__device__ float IsosurfaceHelper::Velocity_Y::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).y;
}

__device__ float IsosurfaceHelper::Velocity_Z::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return  tex3D<float4>(tex, position.x, position.y, position.z).z;
}

__device__ float IsosurfaceHelper::ShearStress::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	float4 dV_dY = tex3D<float4>(tex, position.x, position.y + 0.001 / 2.0f, position.z);
	
	dV_dY -= tex3D<float4>(tex, position.x, position.y - 0.001 / 2.0f, position.z);

	float2 ShearStress =make_float2(dV_dY.x / 0.001f, dV_dY.z / 0.001f);

	return fabsf(sqrtf(dot(ShearStress, ShearStress)));
}


__device__ float4 IsosurfaceHelper::Position::ValueAtXY(cudaTextureObject_t tex, float2 position)
{
	return   tex2D<float4>(tex, position.x, position.y);
}




__device__  float2 IsosurfaceHelper::Position::GradientAtXY_Grid(cudaSurfaceObject_t surf, int2 gridPosition)
{
	float dH_dX = this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x + 1, gridPosition.y));
	float dH_dY = this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x, gridPosition.y + 1));

	dH_dX -= this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x -1, gridPosition.y));
	dH_dY -= this->ValueAtXY_Surface_float(surf, make_int2(gridPosition.x, gridPosition.y -1));

	return make_float2(-dH_dX, -dH_dY);
}

__device__  float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos)
{
	float4 data;
	surf3Dread(&data, surf, gridPos.x * sizeof(float4), gridPos.y, gridPos.z);

	return data;
}

__device__  float2 IsosurfaceHelper::Position::GradientAtXYZ_Grid(cudaSurfaceObject_t surf, int3 gridPosition)
{
	float dH_dX = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).x;
	float dH_dY = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y + 1 , gridPosition.z)).x;

	dH_dX -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y , gridPosition.z)).x;
	dH_dY -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).x;

	

	return make_float2(dH_dX, dH_dY);
}


__device__  float3 GradientAtXYZ_X_Surface(cudaSurfaceObject_t surf, int3 gridPosition)
{
	float dH_dX = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).x;
	float dH_dY = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).x;
	float dH_dZ = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y , gridPosition.z +1)).x;

	dH_dX -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).x;
	dH_dY -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).x;
	dH_dZ -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y , gridPosition.z-1)).x;



	return make_float3(dH_dX, dH_dY, dH_dZ);
}

__device__  float3 GradientAtXYZ_Y_Surface(cudaSurfaceObject_t surf, int3 gridPosition)
{
	float dH_dX = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).y;
	float dH_dY = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).y;
	float dH_dZ = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z + 1)).y;

	dH_dX -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).y;
	dH_dY -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).y;
	dH_dZ -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z - 1)).y;



	return make_float3(dH_dX, dH_dY, dH_dZ);
}

__device__  float3 GradientAtXYZ_Z_Surface(cudaSurfaceObject_t surf, int3 gridPosition)
{
	float dH_dX = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).z;
	float dH_dY = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).z;
	float dH_dZ = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z + 1)).z;

	dH_dX -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).z;
	dH_dY -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).z;
	dH_dZ -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z - 1)).z;



	return make_float3(dH_dX, dH_dY, dH_dZ);
}


__device__  float3 GradientAtXYZ_W_Surface(cudaSurfaceObject_t surf, int3 gridPosition)
{
	float dH_dX = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).w;
	float dH_dY = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y + 1, gridPosition.z)).w;
	float dH_dZ = ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z + 1)).w;

	dH_dX -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).w;
	dH_dY -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y - 1, gridPosition.z)).w;
	dH_dZ -= ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z - 1)).w;



	return make_float3(dH_dX, dH_dY, dH_dZ);
}

__device__ float2 IsosurfaceHelper::Position::GradientFluctuatuionAtXT(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize)
{
	float dH_dX = 0.0f;
	float dH_dY = 0.0f;

	if(gridPosition.x != 0 && gridPosition.x != gridSize.x -1)
	{ 
		dH_dX = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).y;
		dH_dX -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).y;
	}
	else if (gridPosition.x == 0)
	{
		dH_dX = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).y;
		dH_dX -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).y;
		dH_dX = 2 * dH_dX;
	}
	else if (gridPosition.x == gridSize.x - 1)
	{
		dH_dX = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).y;
		dH_dX -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x-1, gridPosition.y, gridPosition.z)).y;
		dH_dX = 2 * dH_dX;
	}

	// Y direction
	if (gridPosition.z != 0 && gridPosition.z != gridSize.z - 1)
	{
		dH_dY = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z + 1)).y;
		dH_dY -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z - 1)).y;
	}
	else if (gridPosition.z == 0)
	{
		dH_dY = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x , gridPosition.y, gridPosition.z+1)).y;
		dH_dY -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).y;
		dH_dY = 2 * dH_dY;
	}
	else if (gridPosition.z == gridSize.z - 1)
	{
		dH_dY = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z)).y;
		dH_dY -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x , gridPosition.y, gridPosition.z-1)).y;
		dH_dY = 2 * dH_dY;
	}



	return make_float2(dH_dX, dH_dY);
}


__device__ float2 IsosurfaceHelper::Position::GradientFluctuatuionAtXZ(cudaSurfaceObject_t surf, int3 gridPosition, int3 gridSize)
{
	float dH_dX = 0.0f;
	float dH_dY = 0.0f;

	if (gridPosition.x % (gridSize.x - 1) != 0 && gridPosition.z % (gridSize.z - 1) != 0)
	{
		dH_dX = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x + 1, gridPosition.y, gridPosition.z)).y;
		dH_dX -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x - 1, gridPosition.y, gridPosition.z)).y;


		dH_dY = this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z + 1)).y;
		dH_dY -= this->ValueAtXYZ_Surface_float4(surf, make_int3(gridPosition.x, gridPosition.y, gridPosition.z - 1)).y;
	}




	return make_float2(dH_dX, dH_dY);
}


__device__  float IsosurfaceHelper::Position::ValueAtXY_Surface_float(cudaSurfaceObject_t tex, int2 gridPos)
{
	float data;
	surf2Dread(&data, tex, gridPos.x *sizeof(float), gridPos.y);

	return data;
};

__device__  float4 IsosurfaceHelper::Position::ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos)
{
	float4 data;
	surf3Dread(&data, surf, gridPos.x * sizeof(float4), gridPos.y,gridPos.z);

	return data;
};

__device__  float4 IsosurfaceHelper::Position::ValueAtXY_Surface_float4(cudaSurfaceObject_t tex, int2 gridPos)
{
	float4 data;
	surf2Dread(&data, tex, gridPos.x * 4 * sizeof(float), gridPos.y);

	return data;
};







__device__  float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z);
}




__device__  float4 IsosurfaceHelper::Observable::ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z);
}


// TODO
__device__  float4 IsosurfaceHelper::Observable::ValueAtXYZ_float4
(
	cudaSurfaceObject_t surf,
	int3 position,
	IsosurfaceHelper::cudaSurfaceAddressMode addressMode ,
	bool interpolation
)
{

	float4 data;

	switch (addressMode)
	{
		case IsosurfaceHelper::cudaSurfaceAddressMode::cudaAddressModeBorder:
		{
			surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);

			break;
		}
		case IsosurfaceHelper::cudaSurfaceAddressMode::cudaAddressModeClamp:
		{
			break;
		}
		case IsosurfaceHelper::cudaSurfaceAddressMode::cudaAddressModeMirror:
		{
			break;
		}
		case IsosurfaceHelper::cudaSurfaceAddressMode::cudaAddressModeWrap:
		{

			break;
		}
	}
	return { -1.0f,-1.0f,-1.0f,-1.0f };
};




__device__ float IsosurfaceHelper::TurbulentDiffusivity::ValueAtXYZ(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z).w;
}

__device__ float IsosurfaceHelper::TurbulentDiffusivity::ValueAtXYZ_avgtemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp)
{

	float temperature = ValueAtXYZ(tex, position);
	float averageTemp = tex1D<float>(avg_temp, position.z);

	float h = 1.0f/ (float)gridSize.z;


	float3 grad = GradientAtGrid(tex, position, gridSize);
	grad.z += (tex1D<float>(avg_temp, position.z + h) - tex1D<float>(avg_temp, position.z - h))/ (2.0f * h);
	float pr = 0.001f;
	float ra = 100000.0f;

	float epsilion_theta = dot(grad, grad) / sqrtf(pr * ra);
	float theta_sqrd = (temperature - averageTemp) * (temperature - averageTemp);

	return  theta_sqrd / epsilion_theta;
}



__device__  float3 IsosurfaceHelper::TurbulentDiffusivity::GradientAtGrid_AvgTemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp)
{
	float3 h = { 1.0f, 1.0f ,1.0f };
	h = 2 * h / gridSize;
	float dV_dX = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x + h.x / 2.0f, position.y, position.z),gridSize,avg_temp);
	float dV_dY = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y + h.y / 2.0f, position.z), gridSize, avg_temp);
	float dV_dZ = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y, position.z + h.z / 2.0f), gridSize, avg_temp);

	dV_dX -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x - h.x / 2.0f, position.y, position.z), gridSize, avg_temp);
	dV_dY -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y - h.y / 2.0f, position.z), gridSize, avg_temp);
	dV_dZ -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y, position.z - h.z / 2.0f), gridSize, avg_temp);

	return { dV_dX / h.x ,dV_dY / h.y, dV_dZ / h.z };
}


__device__ float3 IsosurfaceHelper::TurbulentDiffusivity::binarySearch_avgtemp
(
	cudaTextureObject_t field,
	cudaTextureObject_t average_temp,
	int3& _gridSize,
	float3& _position,
	float3& gridDiameter,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
)
{
	float3 position = _position;
	float3 relative_position = position / gridDiameter;
	float3 samplingStep = _samplingStep * 0.5f;
	bool side = 0; // 1 -> right , 0 -> left
	int counter = 0;



	while (fabsf(ValueAtXYZ_avgtemp(field, relative_position, _gridSize, average_temp) - value) > tolerance&& counter < maxIteration)
	{

		if (ValueAtXYZ_avgtemp(field, relative_position, _gridSize, average_temp) - value > 0)
		{
			if (side)
			{
				samplingStep = 0.5 * samplingStep;
			}
			position = position - samplingStep;
			relative_position = position / gridDiameter;
			side = 0;

		}
		else
		{

			if (!side)
			{
				samplingStep = 0.5 * samplingStep;
			}

			position = position + samplingStep;
			relative_position = position / gridDiameter;
			side = 1;

		}
		counter++;

	}

	return position;

};
