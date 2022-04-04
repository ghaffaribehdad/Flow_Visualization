#pragma once

#include <cuda_runtime.h>
#include "../Cuda/helper_math.h"
#include "../Options/RaycastingOptions.h"

__device__ float4 ValueAtXYZ_Surface_float4(cudaSurfaceObject_t surf, int3 gridPos);
__device__ float4 ValueAtXYZ_float4(cudaTextureObject_t tex, float3 position);
__device__	float3	GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
__device__	float3	GradientAtXYZ_Tex_W(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
__device__	float3	GradientAtXYZ_Tex_X(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
__device__	float3	GradientAtXYZ_Tex_X_Height(cudaTextureObject_t tex, const float3 & position);

namespace FetchTextureSurface
{
	
	struct Measure
	{


		//Return the texel at XYZ of the Texture (Boundaries are controlled by the cudaTextureAddressMode
		__device__	virtual	float 		ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) { return 0; };
		__device__	virtual	float		ValueAtXYZ_Tex_GradientBase(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize) { return 1; };


		__device__	virtual	float		ValueAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position) { return 0; };
		__device__  virtual	float3		GradientAtXYZ_Surf(cudaSurfaceObject_t surf, const int3 & position, const float3 & gridDiameter, const int3 & gridSize);
		__device__			float3		GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
		__device__			float3		GradientAtXYZ_Tex_Absolute(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);
		__device__			float3		GradientAtXYZ_Tex_GradientBase(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize);

		__device__	static	float3		ValueAtXYZ_XYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_XY_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_XZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	static	float2		ValueAtXYZ_YZ_Tex(cudaTextureObject_t tex, const float3 & position);

		__device__	static	float3		ValueAtXYZ_XYZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_XY_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_XZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
		__device__	static	float2		ValueAtXYZ_YZ_Surf(cudaSurfaceObject_t surf, const int3 & position);
	};

	struct Channel_X : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_Y : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_Z : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position);

	};

	struct Channel_W : public Measure
	{
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position) override;

	};


	struct Velocity_Magnitude : public Measure
	{
		// calculates the value of the field at position XYZ
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) override;
		__device__	virtual	float ValueAtXYZ_Surf(cudaSurfaceObject_t tex, const int3 & position) override;


	};


	struct ShearStress : public Measure
	{
		// calculates the value of the field at position XYZ
		__device__	virtual	float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position) override;


	};


	struct Lambda2 : public Measure
	{
		// calculates the value of the field at position XYZ
		__device__	virtual	float ValueAtXYZ_Tex_GradientBase(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiamter, const int3 & gridSize) override;


	};




	struct TurbulentDiffusivity : public Measure
	{
		__device__ float ValueAtXYZ_avgtemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
		__device__ float3 GradientAtGrid_AvgTemp(cudaTextureObject_t tex, float3 position, int3 gridSize, cudaTextureObject_t avg_temp);
		__device__ float3 binarySearch_avgtemp(
			cudaTextureObject_t field,
			cudaTextureObject_t average_temp,
			int3& _gridSize,
			float3& _position,
			float3& gridDiameter,
			float3& _samplingStep,
			float& value,
			float& tolerance,
			int maxIteration
		);
	};

	 




	struct Measure_Jacobian : public Measure
	{

		__device__ static fMat3X3 jacobian(cudaTextureObject_t tex, const float3 & position, const float3 & h);
	};

	struct Vorticity : public Measure_Jacobian
	{
		__device__ float ValueAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position);
	};

}



// The Curiously Recurring Template Pattern (CRTP)
template <class T>
struct Measure
{
	// methods within Base can use template to access members of Derived
	__device__	static	float  ValueAtXYZ(cudaTextureObject_t tex, const float3 & position)
	{
		return T::ValueAtXYZ_derived(tex, position);
	};

	__device__ static float ValueAtXYZ(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		return T::ValueAtXYZ_derived(tex, position, gridDiameter, gridSize);
	};

	__device__ static float3 GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
				
		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += T::ValueAtXYZ_derived(tex, make_float3(position.x + 1, position.y, position.z));
		gradient.y += T::ValueAtXYZ_derived(tex, make_float3(position.x, position.y + 1, position.z));
		gradient.z += T::ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z + 1));

		gradient.x -= T::ValueAtXYZ_derived(tex, make_float3(position.x - 1, position.y, position.z));
		gradient.y -= T::ValueAtXYZ_derived(tex, make_float3(position.x, position.y - 1, position.z));
		gradient.z -= T::ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z - 1));

		return  gradient / (2.0f * h);
	}

};

struct Channel_X : public Measure<Channel_X>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		return tex3D<float4>(tex, position.x, position.y, position.z).x;

	}
};

struct Channel_Y : public Measure<Channel_Y>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		return tex3D<float4>(tex, position.x, position.y, position.z).y;

	}
};

struct Channel_Z : public Measure<Channel_Z>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		return tex3D<float4>(tex, position.x, position.y, position.z).z;

	}
};

struct Channel_W : public Measure<Channel_W>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		return tex3D<float4>(tex, position.x, position.y, position.z).w;

	}
};


struct Velocity_Magnitude : public Measure<Velocity_Magnitude>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		float4 data = tex3D<float4>(tex, position.x, position.y, position.z);
		float3 velocity = make_float3(data.x, data.y, data.z);

		return magnitude(velocity);

	}
};


struct KineticEnergy : public Measure<KineticEnergy>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position)
	{
		float4 data = tex3D<float4>(tex, position.x, position.y, position.z);

	
		return 0.5f * (powf(data.x, 2.0f) + powf(data.y, 2.0f) + powf(data.z, 2.0f));

	}
};


struct ShearStress : public Measure<ShearStress>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		float3 h = gridDiameter / gridSize;

		float4 dV_dY = tex3D<float4>(tex, position.x, position.y + 1, position.z);

		dV_dY -= tex3D<float4>(tex, position.x, position.y - 1, position.z);

		//float2 ShearStress = make_float2(dV_dY.x / (2.0f*h.x), dV_dY.z / (2.0f*h.z));
		
		float3 shear = make_float3(dV_dY.x, dV_dY.y, dV_dY.z) / h;

		//return magnitude(shear);
		
		return shear.x;

		//return fabsf(sqrtf(dot(ShearStress, ShearStress)));

	}

	__device__ static float3 GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += ValueAtXYZ_derived(tex, make_float3(position.x + 1, position.y, position.z), gridDiameter, gridSize);
		gradient.y += ValueAtXYZ_derived(tex, make_float3(position.x, position.y + 1, position.z), gridDiameter, gridSize);
		gradient.z += ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z + 1), gridDiameter, gridSize);

		gradient.x -= ValueAtXYZ_derived(tex, make_float3(position.x - 1, position.y, position.z), gridDiameter, gridSize);
		gradient.y -= ValueAtXYZ_derived(tex, make_float3(position.x, position.y - 1, position.z), gridDiameter, gridSize);
		gradient.z -= ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z - 1), gridDiameter, gridSize);

		return  gradient / (2.0f * h);
	}
};


struct Lambda2 : public Measure<Lambda2>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		return lambda2(tex, position, gridDiameter, gridSize);
	}

	__device__ static float3 GradientAtXYZ_Tex(cudaTextureObject_t tex, const float3 & position, const float3 & gridDiameter, const int3 & gridSize) 
	{
		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += ValueAtXYZ_derived(tex, make_float3(position.x + 1, position.y, position.z), gridDiameter, gridSize);
		gradient.y += ValueAtXYZ_derived(tex, make_float3(position.x, position.y + 1, position.z), gridDiameter, gridSize);
		gradient.z += ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z + 1), gridDiameter, gridSize);

		gradient.x -= ValueAtXYZ_derived(tex, make_float3(position.x - 1, position.y, position.z), gridDiameter, gridSize);
		gradient.y -= ValueAtXYZ_derived(tex, make_float3(position.x, position.y - 1, position.z), gridDiameter, gridSize);
		gradient.z -= ValueAtXYZ_derived(tex, make_float3(position.x, position.y, position.z - 1), gridDiameter, gridSize);

		return  gradient / (2.0f * h);
	}
};



__global__ void mipmapped(cudaTextureObject_t tex, cudaSurfaceObject_t surf, int3 gridSize, int z);

