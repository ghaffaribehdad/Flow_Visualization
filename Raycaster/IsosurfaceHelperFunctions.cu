
#include "IsosurfaceHelperFunctions.h"
#include "..//Cuda/helper_math.h"
#include "BoundingBox.h"
#include "..//Cuda/CudaHelperFunctions.h"


//__device__ float3 FetchTextureSurface::Measure::ValueAtXYZ_XYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
//	return make_float3(data.x, data.y, data.z);
//
//}
//
//
//__device__  float FetchTextureSurface::Channel_X::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	return tex3D<float4>(tex, position.x, position.y, position.z).x;
//
//}
//
//__device__  float FetchTextureSurface::Difference::ValueAtXYZ_Tex(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
//{
//	return (tex3D<float4>(tex1, position.x, position.y, position.z).x - tex3D<float4>(tex0, position.x, position.y, position.z).x);
//
//}
//
//__device__  float FetchTextureSurface::Channel_Y::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	return tex3D<float4>(tex, position.x, position.y, position.z).y;
//
//}
//__device__  float FetchTextureSurface::Channel_Z::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	return tex3D<float4>(tex, position.x, position.y, position.z).z;
//
//}
//__device__  float FetchTextureSurface::Channel_W::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	return tex3D<float4>(tex, position.x, position.y, position.z).w;
//
//}
//
//
//__device__ float2 FetchTextureSurface::Measure::ValueAtXYZ_XY_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//
//	float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
//	return make_float2(data.x, data.y);
//
//}
//
//__device__ float2 FetchTextureSurface::Measure::ValueAtXYZ_XZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//
//	float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
//	return make_float2(data.x, data.z);
//
//}
//
//__device__ float2 FetchTextureSurface::Measure::ValueAtXYZ_YZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//
//	float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
//	return make_float2(data.y, data.z);
//
//}
//
//
//
//__device__  float3 FetchTextureSurface::Measure::ValueAtXYZ_XYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return make_float3(data.x,data.y,data.z);
//}
//
//
//__device__  float2 FetchTextureSurface::Measure::ValueAtXYZ_XY_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return make_float2(data.x, data.y);
//}
//
//__device__  float2 FetchTextureSurface::Measure::ValueAtXYZ_XZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return make_float2(data.x, data.z);
//}
//
//__device__  float2 FetchTextureSurface::Measure::ValueAtXYZ_YZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return make_float2(data.y, data.z);
//}
//
//
//
//__device__ float3 FetchTextureSurface::Measure::GradientAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position,const float3 & gridDiamter, const int3 & gridSize)
//{
//	float3 h = gridDiamter / gridSize;
//	float3 gradient = { 0,0,0 };
//
//	if (position.x == 0)
//	{
//		gradient.x += ValueAtXYZ_Surf(surf, make_int3(position.x + 1, position.y, position.z));
//		gradient.x -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.x = gradient.x / h.x;
//	}
//	else if (position.x == gridSize.x - 1)
//	{
//		gradient.x += ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.x -= ValueAtXYZ_Surf(surf, make_int3(position.x-1, position.y, position.z));
//		gradient.x = gradient.x / h.x;
//	}
//	else
//	{
//		gradient.x += ValueAtXYZ_Surf(surf, make_int3(position.x + 1, position.y, position.z));
//		gradient.x -= ValueAtXYZ_Surf(surf, make_int3(position.x - 1, position.y, position.z));
//		gradient.x = gradient.x / (2.0f * h.x);
//	}
//
//	if (position.y == 0)
//	{
//		gradient.y += ValueAtXYZ_Surf(surf, make_int3(position.x , position.y + 1, position.z));
//		gradient.y -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.y = gradient.y / h.y;
//	}
//	else if (position.y == gridSize.y - 1)
//	{
//		gradient.y += ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.y -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y - 1, position.z));
//		gradient.y = gradient.y / h.y;
//	}
//	else
//	{
//		gradient.y += ValueAtXYZ_Surf(surf, make_int3(position.x, position.y + 1, position.z));
//		gradient.y -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y - 1, position.z));
//		gradient.y = gradient.y / (2.0f *h.y);
//
//	}
//
//	if (position.z == 0)
//	{
//		gradient.z += ValueAtXYZ_Surf(surf, make_int3(position.x , position.y, position.z + 1));
//		gradient.z -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.z = gradient.z / h.z;
//	}
//	else if (position.z == gridSize.z - 1)
//	{
//		gradient.z += ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z));
//		gradient.z -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z -1));
//		gradient.z = gradient.z / h.z;
//	}
//	else
//	{
//		gradient.z += ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z+1));
//		gradient.z -= ValueAtXYZ_Surf(surf, make_int3(position.x, position.y, position.z-1));
//		gradient.z = gradient.z / (2.0f*h.z);
//
//	}
//
//	return gradient;
//}
//
//
//__device__ float3 FetchTextureSurface::Measure::GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
//{
//	float3 h = gridDiameter / gridSize;
//	float3 gradient = { 0,0,0 };
//
//
//	gradient.x += ValueAtXYZ_Tex(tex, make_float3(position.x + 1, position.y, position.z));
//	gradient.y += ValueAtXYZ_Tex(tex, make_float3(position.x, position.y + 1, position.z));
//	gradient.z += ValueAtXYZ_Tex(tex, make_float3(position.x, position.y, position.z + 1));
//
//	gradient.x -= ValueAtXYZ_Tex(tex, make_float3(position.x - 1, position.y, position.z));
//	gradient.y -= ValueAtXYZ_Tex(tex, make_float3(position.x, position.y - 1, position.z));
//	gradient.z -= ValueAtXYZ_Tex(tex, make_float3(position.x, position.y, position.z - 1));
//
//
//	return  gradient /(2.0f * h);
//
//}
//
//__device__ float3 FetchTextureSurface::Measure::GradientAtXYZ_Tex_Absolute(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
//{
//	float3 h = gridDiameter / gridSize;
//	float3 gradient = { 0,0,0 };
//
//
//	gradient.x += fabs(ValueAtXYZ_Tex(tex, make_float3(position.x + 1, position.y, position.z)));
//	gradient.y += fabs(ValueAtXYZ_Tex(tex, make_float3(position.x, position.y + 1, position.z)));
//	gradient.z += fabs(ValueAtXYZ_Tex(tex, make_float3(position.x, position.y, position.z + 1)));
//
//	gradient.x -= fabs(ValueAtXYZ_Tex(tex, make_float3(position.x - 1, position.y, position.z)));
//	gradient.y -= fabs(ValueAtXYZ_Tex(tex, make_float3(position.x, position.y - 1, position.z)));
//	gradient.z -= fabs(ValueAtXYZ_Tex(tex, make_float3(position.x, position.y, position.z - 1)));
//
//	return  gradient / (2.0f * h);
//
//}
//
//
//
//__device__ float3 FetchTextureSurface::Measure::GradientAtXYZ_Tex_GradientBase(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize)
//{
//	float3 h = gridDiamter / gridSize;
//	float3 gradient = { 0,0,0 };
//
//
//	gradient.x += ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x + 1, position.y, position.z), gridDiamter,gridSize);
//	gradient.y += ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x, position.y + 1, position.z), gridDiamter, gridSize);
//	gradient.z += ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x, position.y, position.z + 1), gridDiamter, gridSize);
//
//	gradient.x -= ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x - 1, position.y, position.z), gridDiamter, gridSize);
//	gradient.y -= ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x, position.y - 1, position.z), gridDiamter, gridSize);
//	gradient.z -= ValueAtXYZ_Tex_GradientBase(tex, make_float3(position.x, position.y, position.z - 1), gridDiamter, gridSize);
//
//
//	return  gradient / (2.0f * h);
//
//}



__device__ float3 GradientAtXYZ_Tex_W(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize)
{
	float3 h = gridDiamter / gridSize;
	float3 gradient = { 0,0,0 };

	
	gradient.x += tex3D<float4>(tex, position.x+1, position.y, position.z).w;
	gradient.y += tex3D<float4>(tex, position.x , position.y +1, position.z).w;
	gradient.z += tex3D<float4>(tex, position.x , position.y, position.z +1).w;

	gradient.x -= tex3D<float4>(tex, position.x-1, position.y, position.z).w;
	gradient.y -= tex3D<float4>(tex, position.x , position.y-1, position.z).w;
	gradient.z -= tex3D<float4>(tex, position.x , position.y, position.z-1).w;


	return  gradient / (2.0f * h);

}


__device__ float3 GradientAtXYZ_Tex_X(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize)
{
	float3 h = gridDiamter / gridSize;
	float3 gradient = { 0,0,0 };

	gradient.x += tex3D<float4>(tex, position.x + 1, position.y, position.z).x;
	gradient.y += tex3D<float4>(tex, position.x, position.y + 1, position.z).x;
	gradient.z += tex3D<float4>(tex, position.x, position.y, position.z + 1).x;

	gradient.x -= tex3D<float4>(tex, position.x - 1, position.y, position.z).x;
	gradient.y -= tex3D<float4>(tex, position.x, position.y - 1, position.z).x;
	gradient.z -= tex3D<float4>(tex, position.x, position.y, position.z - 1).x;


	return  gradient / (2.0f * h);

}

__device__ float3 GradientAtXYZ_Tex_X_Height(cudaTextureObject_t tex, const float3 & position)
{
	float3 gradient = { 0,0,0 };


	gradient.x += tex3D<float4>(tex, position.x + 1, position.y, position.z).x;
	gradient.y += tex3D<float4>(tex, position.x, position.y + 1, position.z).x;

	gradient.x -= tex3D<float4>(tex, position.x - 1, position.y, position.z).x;
	gradient.y -= tex3D<float4>(tex, position.x, position.y - 1, position.z).x;


	return  gradient;

}


__device__ float3 GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize)
{
	float3 h = gridDiamter / gridSize;
	float3 gradient = { 0,0,0 };


	gradient.x += tex3D<float>(tex, position.x + 1, position.y, position.z);
	gradient.y += tex3D<float>(tex, position.x, position.y + 1, position.z);
	gradient.z += tex3D<float>(tex, position.x, position.y, position.z + 1);

	gradient.x -= tex3D<float>(tex, position.x - 1, position.y, position.z);
	gradient.y -= tex3D<float>(tex, position.x, position.y - 1, position.z);
	gradient.z -= tex3D<float>(tex, position.x, position.y, position.z - 1);


	return  gradient / (2.0f * h);

}


//__device__  float FetchTextureSurface::Channel_X::ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return data.x;
//}
//
//
//__device__  float FetchTextureSurface::Channel_Y::ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return data.y;
//}
//
//__device__  float FetchTextureSurface::Channel_Z::ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return data.z;
//}
//
//__device__  float FetchTextureSurface::Channel_W::ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//
//	return data.w;
//}
//
//__device__  float FetchTextureSurface::Velocity_Magnitude::ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position)
//{
//	float4 data;
//	surf3Dread(&data, surf, position.x * sizeof(float4), position.y, position.z);
//	float3 velocity = make_float3(data.x, data.y, data.z);
//
//	return magnitude(velocity);
//}




//__device__ float FetchTextureSurface::Velocity_Magnitude::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//
//	float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
//	float3 velocity = make_float3(data.x, data.y, data.z);
//
//	return magnitude(velocity);
//}
//
//__device__  float3 GradientAtGrid_X(cudaTextureObject_t tex, float3 position, int3 gridSize)
//{
//	float3 h = { 1.0f, 1.0f ,1.0f };
//	h = h / gridSize;
//	float dV_dX = ValueAtXYZ_float4(tex, make_float3(position.x + 1, position.y, position.z)).x;
//	float dV_dY = ValueAtXYZ_float4(tex, make_float3(position.x, position.y + 1, position.z)).x;
//	float dV_dZ = ValueAtXYZ_float4(tex, make_float3(position.x, position.y, position.z + 1)).x;
//
//	dV_dX -= ValueAtXYZ_float4(tex, make_float3(position.x - 1, position.y, position.z)).x;
//	dV_dY -= ValueAtXYZ_float4(tex, make_float3(position.x, position.y - 1, position.z)).x;
//	dV_dZ -= ValueAtXYZ_float4(tex, make_float3(position.x, position.y, position.z - 1)).x;
//
//	return { dV_dX / h.x ,dV_dY / h.y, dV_dZ / h.z };
//}



//__device__ float FetchTextureSurface::ShearStress::ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position)
//{
//	float4 dV_dY = tex3D<float4>(tex, position.x, position.y + 1, position.z);
//	
//	dV_dY -= tex3D<float4>(tex, position.x, position.y - 1, position.z);
//
//	float2 ShearStress =make_float2(dV_dY.x / 2.0f, dV_dY.z / 2.0f);
//
//	return fabsf(sqrtf(dot(ShearStress, ShearStress)));
//}






__device__  float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos)
{
	float4 data;
	surf3Dread(&data, surf, gridPos.x * sizeof(float4), gridPos.y, gridPos.z);

	return data;
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


__device__  float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position)
{
	return tex3D<float4>(tex, position.x, position.y, position.z);
}


//__device__ float FetchTextureSurface::TurbulentDiffusivity::ValueAtXYZ_avgtemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp)
//{
//
//	//float temperature = ValueAtXYZ(tex, position);
//	//float averageTemp = tex1D<float>(avg_temp, position.z);
//
//	//float h = 1.0f/ (float)gridSize.z;
//
//
//	//float3 grad = GradientAtGrid(tex, position, gridSize);
//	//grad.z += (tex1D<float>(avg_temp, position.z + h) - tex1D<float>(avg_temp, position.z - h))/ (2.0f * h);
//	//float pr = 0.001f;
//	//float ra = 100000.0f;
//
//	//float epsilion_theta = dot(grad, grad) / sqrtf(pr * ra);
//	//float theta_sqrd = (temperature - averageTemp) * (temperature - averageTemp);
//
//	//return  theta_sqrd / epsilion_theta;
//	return 0;
//}


//
//__device__  float3 FetchTextureSurface::TurbulentDiffusivity::GradientAtGrid_AvgTemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp)
//{
//	float3 h = { 1.0f, 1.0f ,1.0f };
//	h = 2 * h / gridSize;
//	float dV_dX = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x + h.x / 2.0f, position.y, position.z),gridSize,avg_temp);
//	float dV_dY = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y + h.y / 2.0f, position.z), gridSize, avg_temp);
//	float dV_dZ = this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y, position.z + h.z / 2.0f), gridSize, avg_temp);
//
//	dV_dX -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x - h.x / 2.0f, position.y, position.z), gridSize, avg_temp);
//	dV_dY -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y - h.y / 2.0f, position.z), gridSize, avg_temp);
//	dV_dZ -= this->ValueAtXYZ_avgtemp(tex, make_float3(position.x, position.y, position.z - h.z / 2.0f), gridSize, avg_temp);
//
//	return { dV_dX / h.x ,dV_dY / h.y, dV_dZ / h.z };
//}


//__device__ float3 FetchTextureSurface::TurbulentDiffusivity::binarySearch_avgtemp
//(
//	cudaTextureObject_t field,
//	cudaTextureObject_t average_temp,
//	int3& _gridSize,
//	float3& _position,
//	float3& gridDiameter,
//	float3& _samplingStep,
//	float& value,
//	float& tolerance,
//	int maxIteration
//)
//{
//	float3 position = _position;
//	float3 relative_position = position / gridDiameter;
//	float3 samplingStep = _samplingStep * 0.5f;
//	bool side = 0; // 1 -> right , 0 -> left
//	int counter = 0;
//
//
//
//	while (fabsf(ValueAtXYZ_avgtemp(field, relative_position, _gridSize, average_temp) - value) > tolerance&& counter < maxIteration)
//	{
//
//		if (ValueAtXYZ_avgtemp(field, relative_position, _gridSize, average_temp) - value > 0)
//		{
//			if (side)
//			{
//				samplingStep = 0.5 * samplingStep;
//			}
//			position = position - samplingStep;
//			relative_position = position / gridDiameter;
//			side = 0;
//
//		}
//		else
//		{
//
//			if (!side)
//			{
//				samplingStep = 0.5 * samplingStep;
//			}
//
//			position = position + samplingStep;
//			relative_position = position / gridDiameter;
//			side = 1;
//
//		}
//		counter++;
//
//	}
//
//	return position;
//
//};



//__device__ fMat3X3 FetchTextureSurface::Measure_Jacobian::jacobian(cudaTextureObject_t tex, const float3 & position, const float3 & h)
//{
//	fMat3X3 jac = { 0,0,0,0,0,0,0,0,0 };
//
//	jac.r1 = Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x + 1, position.y, position.z));
//	jac.r1 -= Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x - 1, position.y, position.z));
//
//	jac.r2 = Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x, position.y +1, position.z));
//	jac.r2 -= Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x, position.y-1, position.z));
//
//	jac.r3 = Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x , position.y, position.z+1));
//	jac.r3 -= Measure::ValueAtXYZ_XYZ_Tex(tex, make_float3(position.x , position.y, position.z-1));
//
//	// This would give us the Jacobian Matrix
//	jac.r1 = jac.r1 / (2 * h.x);
//	jac.r2 = jac.r2 / (2 * h.y);
//	jac.r2 = jac.r3 / (2 * h.z);
//
//	return jac;
//	;
//}

//
// gridSize: size of the mipmapped field
__global__ void mipmapped(cudaTextureObject_t tex, cudaSurfaceObject_t surf, int3 gridSize, int z)
{
	int index = CUDA_INDEX;

	if (index < gridSize.x * gridSize.y)
	{

		
		float3 surf_position = { 0,0,0 };
		surf_position.x = index / gridSize.y;
		surf_position.y = index - surf_position.x * gridSize.y;
		surf_position.z = (float)z;

		float3 tex_position = 2 * surf_position;
		tex_position = tex_position+make_float3(0.5f, 0.5f, 0.5f);


		float4 avg_value = { 0,0,0,0};
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 0, tex_position.y + 0, tex_position.z + 0);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 0, tex_position.y + 0, tex_position.z + 1);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 0, tex_position.y + 1, tex_position.z + 0);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 0, tex_position.y + 1, tex_position.z + 1);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 1, tex_position.y + 0, tex_position.z + 0);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 1, tex_position.y + 0, tex_position.z + 1);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 1, tex_position.y + 1, tex_position.z + 0);
		avg_value = avg_value + tex3D<float4>(tex, tex_position.x + 1, tex_position.y + 1, tex_position.z + 1);
		avg_value = avg_value / 8.0f;

		// write it back to surface
		surf3Dwrite(avg_value, surf, 4 * sizeof(float) * surf_position.x, surf_position.y, surf_position.z);
	}
	
}

