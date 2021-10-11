
#pragma once
#include "cuda_runtime.h"
#include "../ErrorLogger/ErrorLogger.h"
#include "texture_fetch_functions.h"
#include "../Options/SolverOptions.h"

class VolumeTexture3D
{


private:

	bool internalArray = false;
	bool initialized = false;
public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}


	void setArray(cudaArray_t& _cuArray_velocity)
	{
		this->cuArray_velocity = _cuArray_velocity;
	}



	bool initialize
	(
		const int3 & dimension,
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
		);

	bool initialize_devicePointer
	(
		const int3 & dimension,
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);

	// If the cuda array has been set directly 
	bool initialize_array
	(
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear

	);

	void release();
	void destroyTexture()
	{
		gpuErrchk(cudaDestroyTextureObject(this->t_field));
	}
	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}
	

private:

	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity = NULL;
	float* h_field = nullptr;
	float3 gridDiameter;



};






class VolumeTexture2D
{

public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}

	void setArray(cudaArray_t& _cuArray_velocity)
	{
		this->cuArray_velocity = _cuArray_velocity;
	}

	bool initialize
	(
		const int2 & gridSize,
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);

	// If the cuda array has been set directly 
	bool initialize_array
	(
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);


	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}


private:


	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;

	float* h_field = nullptr;

	float3 gridDiameter;



};




class VolumeTexture1D
{

public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}

	void setArray(cudaArray_t& _cuArray_velocity)
	{
		this->cuArray_velocity = _cuArray_velocity;
	}


	bool initialize
	(
		size_t width,
		bool normalizedCoords				= false,
		cudaTextureAddressMode addressMode_z = cudaAddressModeBorder,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);

	bool initialize_array
	(
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_z = cudaAddressModeBorder,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);



	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}


private:



	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;

	float* h_field = nullptr;

	float gridDiameter;



};




template <typename T>
class VolumeTexture3D_T
{


private:
	bool internalArray = false;

public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}


	void setArray(cudaArray_t& _cuArray_velocity)
	{
		this->cuArray_velocity = _cuArray_velocity;
	}



	bool initialize
	(
		const int3 & dimension,
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);

	// If the cuda array has been set directly 

	bool initialize_array
	(
		bool normalizedCoords = false,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear

	);

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}


private:

	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;
	float* h_field = nullptr;
	float3 gridDiameter;



};




//class VolumeTexture3D_Mipmap
//{
//
//public:
//
//	// setter functions
//	void setField(float* _h_field)
//	{
//		this->h_field = _h_field;
//	}
//
//
//	void setArray(cudaTextureObject_t& _cudaMMPArray)
//	{
//		this->cudaMMPArray = _cudaMMPArray;
//	}
//
//
//
//	bool initialize
//	(
//		const int3 & dimension,
//		bool normalizedCoords = false,
//		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
//		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
//		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
//		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
//	);
//
//	bool initialize_devicePointer
//	(
//		const int3 & dimension,
//		bool normalizedCoords = false,
//		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
//		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
//		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
//		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
//	);
//
//	// If the cuda array has been set directly 
//	bool initialize_array
//	(
//		bool normalizedCoords = false,
//		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
//		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
//		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
//		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
//
//	);
//
//	void release();
//	void destroyTexture()
//	{
//		gpuErrchk(cudaDestroyTextureObject(this->t_field));
//	}
//	cudaTextureObject_t getTexture()
//	{
//		return this->t_field;
//	}
//
//
//private:
//	
//	cudaTextureObject_t t_field = NULL;
//	cudaMipmappedArray_t cudaMMPArray = NULL;
//	cudaArray_t cuArray_velocity = NULL;
//	float* h_field = nullptr;
//	float3 gridDiameter;
//	unsigned int nlevel = 0;
//
//
//
//};
