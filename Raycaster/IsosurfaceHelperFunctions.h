#pragma once

#include <cuda_runtime.h>
#include "../Cuda/helper_math.h"
#include "../Options/RaycastingOptions.h"
#include "../Raycaster/Raycasting_Helper.h"




__device__ inline float FTLE_pathSpaceTime(cudaTextureObject_t Field, float3 position, float3 gridDimension, int3 gridSize, float sigma, int3 offset0, int3 offset1) {


	//float3 p_t0_pos = world2TexReshape(make_float3(position.x , position.y, position.z), gridDimension, gridSize, offset0);
	//float3 p_t1_pos = world2TexReshape(make_float3(position.x , position.y, position.z), gridDimension, gridSize, offset1);
	//printf("actual position (%3f,%3f,%3f)\n tex position is at (%3f,%3f,%3f)\n", position.x, position.y, position.z,p_t1_pos.x, p_t1_pos.y, p_t1_pos.z);

	float3 p_t1_pos = position;

	float3 p1 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x + sigma, p_t1_pos.y, p_t1_pos.z));
	float3 p2 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x - sigma, p_t1_pos.y, p_t1_pos.z));
	float3 p3 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x, p_t1_pos.y + sigma, p_t1_pos.z));
	float3 p4 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x, p_t1_pos.y - sigma, p_t1_pos.z));
	float3 p5 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x, p_t1_pos.y, p_t1_pos.z + sigma));
	float3 p6 = FLOAT4XYZ(tex3D<float4>(Field, p_t1_pos.x, p_t1_pos.y, p_t1_pos.z - sigma));
	
	fMat3X3 d_Flowmap(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

	d_Flowmap.r1.x = (p1.x - p2.x);
	d_Flowmap.r2.x = (p3.x - p4.x);
	d_Flowmap.r3.x = (p5.x - p6.x);

	d_Flowmap.r1.y = (p1.y - p2.y);
	d_Flowmap.r2.y = (p3.y - p4.y);
	d_Flowmap.r3.y = (p5.y - p6.y);

	d_Flowmap.r1.z = (p1.z - p2.z);
	d_Flowmap.r2.z = (p3.z - p4.z);
	d_Flowmap.r3.z = (p5.z - p6.z);

	fMat3X3 td_Flowmap = transpose(d_Flowmap);
	fMat3X3 delta = mult(d_Flowmap, td_Flowmap);

	float lambda_max = eigenValueMax(delta);
	
	return logf((sqrtf(lambda_max)));
	
};

// The Curiously Recurring Template Pattern (CRTP)
template <class T>
struct Measure
{
	// methods within Base can use template to access members of Derived
	__device__	static	float  ValueAtXYZ(cudaTextureObject_t tex0, const float3 & position)
	{
		return T::ValueAtXYZ_derived(tex0, position);
	};


	__device__ static float ValueAtXYZ(cudaTextureObject_t tex0, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		return T::ValueAtXYZ_derived(tex0, position, gridDiameter, gridSize);
	};


	__device__ static float ValueAtXYZ(cudaTextureObject_t tex0, const float3 & position, const float3 & gridDiameter, const int3 & gridSize, const float & sigma, const int3& offset0, const int3& offset1)
	{
		return T::ValueAtXYZ_derived(tex0, position, gridDiameter, gridSize,sigma,offset0,offset1);
	};

	__device__	static	float  ValueAtXYZ_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		return T::ValueAtXYZ_derived_double(tex0, tex1, position);
	};


	__device__ static float ValueAtXYZ_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
		return T::ValueAtXYZ_derived_double(tex0, tex1, position, gridDiameter, gridSize);
	};

	__device__ static float3 GradientAtXYZ_Tex(cudaTextureObject_t tex0, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{
				
		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += T::ValueAtXYZ_derived(tex0, make_float3(position.x + 1, position.y, position.z));
		gradient.y += T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y + 1, position.z));
		gradient.z += T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y, position.z + 1));

		gradient.x -= T::ValueAtXYZ_derived(tex0, make_float3(position.x - 1, position.y, position.z));
		gradient.y -= T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y - 1, position.z));
		gradient.z -= T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y, position.z - 1));

		return  gradient / (2.0f * h);
	}

	__device__ static float3 GradientAtXYZ_Tex(cudaTextureObject_t tex0, const float3 & position, const float3 & gridDiameter, const int3 & gridSize, const float & sigma, const int3& offset0, const int3& offset1)
	{

		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += T::ValueAtXYZ_derived(tex0, make_float3(position.x + 1, position.y, position.z), gridDiameter, gridSize, sigma, offset0, offset1);
		gradient.y += T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y + 1, position.z), gridDiameter, gridSize, sigma, offset0, offset1);
		gradient.z += T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y, position.z + 1), gridDiameter, gridSize, sigma, offset0, offset1);

		gradient.x -= T::ValueAtXYZ_derived(tex0, make_float3(position.x - 1, position.y, position.z), gridDiameter, gridSize, sigma, offset0, offset1);
		gradient.y -= T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y - 1, position.z), gridDiameter, gridSize, sigma, offset0, offset1);
		gradient.z -= T::ValueAtXYZ_derived(tex0, make_float3(position.x, position.y, position.z - 1), gridDiameter, gridSize, sigma, offset0, offset1);

		return  gradient / (2.0f * h);
	}



	__device__ static float3 GradientAtXYZ_Tex_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position, const float3 & gridDiameter, const int3 & gridSize)
	{

		float3 h = gridDiameter / gridSize;
		float3 gradient = { 0,0,0 };


		gradient.x += T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x + 1, position.y, position.z));
		gradient.y += T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x, position.y + 1, position.z));
		gradient.z += T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x, position.y, position.z + 1));

		gradient.x -= T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x - 1, position.y, position.z));
		gradient.y -= T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x, position.y - 1, position.z));
		gradient.z -= T::ValueAtXYZ_derived_double(tex0, tex1 , make_float3(position.x, position.y, position.z - 1));

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

struct Difference_X : public Measure<Difference_X>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		return fabs(tex3D<float4>(tex1, position.x, position.y, position.z).x - tex3D<float4>(tex0, position.x, position.y, position.z).x);

	}
};

struct Difference_Y : public Measure<Difference_Y>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		return fabs(tex3D<float4>(tex1, position.x, position.y, position.z).y - tex3D<float4>(tex0, position.x, position.y, position.z).y);

	}
};

struct Difference_Z : public Measure<Difference_Z>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		return fabs(tex3D<float4>(tex1, position.x, position.y, position.z).z - tex3D<float4>(tex0, position.x, position.y, position.z).z);

	}
};

struct Difference_W : public Measure<Difference_W>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		return fabs(tex3D<float4>(tex1, position.x, position.y, position.z).w - tex3D<float4>(tex0, position.x, position.y, position.z).w);

	}
};

struct Average_x_double : public Measure<Average_x_double>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		float value = tex3D<float4>(tex1, position.x, position.y, position.z).x + tex3D<float4>(tex0, position.x, position.y, position.z).x;
		return value* 0.5f;

	}
};

struct Max_x_double : public Measure<Max_x_double>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position)
	{
		float v0 = tex3D<float4>(tex0, position.x, position.y, position.z).x;
		float v1 = tex3D<float4>(tex1, position.x, position.y, position.z).x;
		if (v0 < v1)
			return v0;
		else
			return v1;

	}
};

struct Min_x_double: public Measure<Min_x_double>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position) 
	{
		float v0 = tex3D<float4>(tex0, position.x, position.y, position.z).x;
		float v1 = tex3D<float4>(tex1, position.x, position.y, position.z).x;
		if (v0 > v1)
			return v0;
		else
			return v1;

	}
};


struct normalCurve_x_double : public Measure<normalCurve_x_double>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position) 
	{
		float value = tex3D<float4>(tex1, position.x, position.y, position.z).x + tex3D<float4>(tex0, position.x, position.y, position.z).x;
		return value * 0.5f;

	}
};

struct Importance_diff_less : public Measure<Importance_diff_less>
{
	__device__ static float ValueAtXYZ_derived_double(cudaTextureObject_t tex0, cudaTextureObject_t tex1, const float3 & position) 
	{
		float value1 = tex3D<float4>(tex1, position.x, position.y, position.z).x;
		float value0 = tex3D<float4>(tex0, position.x, position.y, position.z).x;
		float diff = fabs(value1 - value0);
		return diff * 0.5f;
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


struct FTLE : public Measure<FTLE>
{
	__device__ static float ValueAtXYZ_derived(cudaTextureObject_t tex, const float3 & position,const float3 & gridDiameter, const int3 & gridSize, const float & sigma ,const int3& offset0, const int3& offset1)
	{
		return FTLE_pathSpaceTime(tex, position, gridDiameter, gridSize, sigma, offset0, offset1);
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
		
		float3 shear = make_float3(dV_dY.x, dV_dY.y, dV_dY.z) / (2.0f * h);

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

