#pragma once

#include <d3d11.h>
#include "CudaDevice.h"
#include <wrl/client.h>
#include "../BinaryStream/BinaryReader.h"
#include "../BinaryStream/BinaryWriter.h"
#include "../ErrorLogger.h"
#include <DirectXMath.h>
#include "../Particle.cuh"
#include "../VelocityField.h"
#include "../SolverOptions.h"

enum SeedingPattern
{
	SEED_RANDOM = 0,
	//SEED_REGULAR,
	//SEED_FILE,
};

enum IntegrationMethod
{
	EULER_METHOD = 0,
	//MODIFIED_EULER,
	//RK4_METHOD,
	//RK5_METHOD,
};

enum InterpolationMethod
{
	Linear,
};

class CUDASolver
{
public:
	CUDASolver();


	bool Initialize
	(
		SeedingPattern _seedingPattern,
		IntegrationMethod _integrationMethod,
		InterpolationMethod _interpolationMethod,
		SolverOptions _solverOptions
	);

	// Solve must be defined in the derived classes
	virtual bool solve()
	{
		return true;
	}

protected:
	
	// Read and copy file into memeory and save a pointer to it
	bool ReadField(std::vector<char>* p_vec_buffer, std::string fileName);

	// Upload Field to GPU and returns a pointer to the data on GPU
	template <class T>
	T * UploadToGPU(T * host_Data, size_t _size)
	{
		T* device_data;

		gpuErrchk(cudaMalloc((void**)& device_data, _size));

		gpuErrchk(cudaMemcpy(device_data, host_Data, _size, cudaMemcpyHostToDevice));

		return  device_data;
	}



	// Particle tracing parameters
	Particle* h_particles;
	SeedingPattern m_seedingPattern;
	IntegrationMethod m_intergrationMehotd;
	InterpolationMethod m_interpolationMethod;
	unsigned int m_initialTimestep;

	// Solver Parameters
	SolverOptions solverOptions;

	void InitializeParticles(int& particle_count, float3& gridDimenstions, SeedingPattern seedingPattern);

	// A COM pointer to the vector Field
	Microsoft::WRL::ComPtr<ID3D11Texture3D> m_resultTexture;

	CudaDevice cudaDevice;
	bool GetDevice();
};