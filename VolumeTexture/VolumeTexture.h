
#pragma once
#include "cuda_runtime.h"
#include "../ErrorLogger/ErrorLogger.h"
#include "texture_fetch_functions.h"
#include "../Options/SolverOptions.h"

class VolumeTexture3D
{

public:

	// setter functions
	void setField(float* _h_field)
	{
		this->h_field = _h_field;
	}

	void setSolverOptions(SolverOptions* _solverOptions)
	{
		this->solverOptions = _solverOptions;
	}

	void setArray(cudaArray_t& _cuArray_velocity)
	{
		this->cuArray_velocity = _cuArray_velocity;
	}


	// Create a texture and populate it with h_field
	// Address modes can be set for X,y,z
	bool initialize
	(
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear

	);

	bool initialize
	(
		int3 dimension,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
		);

	bool initialize_array
	(
		int3 dimension,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap

	);

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}
	

private:

	SolverOptions * solverOptions = nullptr;

	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;

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

	// Create a texture and populate it with h_field
	// Address modes can be set for X,y,z
	bool initialize
	(
		size_t width,
		size_t height,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap
	);

	bool initialize_array
	(
		size_t width,
		size_t height,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap

	);


	bool initialize
	(
		size_t width,
		size_t height,
		cudaTextureAddressMode addressMode_x = cudaAddressModeWrap,
		cudaTextureAddressMode addressMode_y = cudaAddressModeBorder,
		cudaTextureFilterMode _cudaTextureFilterMode = cudaFilterModeLinear
	);

	void setSolverOptions(SolverOptions* _solverOptions)
	{
		this->solverOptions = _solverOptions;
	}

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}


private:

	SolverOptions* solverOptions = nullptr;

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

	// Create a texture and populate it with h_field
	// Address modes can be set for X,y,z
	bool initialize
	(
		size_t width,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap
	);

	bool initialize_array
	(
		size_t width,
		cudaTextureAddressMode addressMode_z = cudaAddressModeWrap

	);


	void setSolverOptions(SolverOptions* _solverOptions)
	{
		this->solverOptions = _solverOptions;
	}

	void release();

	cudaTextureObject_t getTexture()
	{
		return this->t_field;
	}


private:

	SolverOptions* solverOptions = nullptr;

	cudaTextureObject_t t_field;
	cudaArray_t cuArray_velocity;

	float* h_field = nullptr;

	float gridDiameter;



};
