#pragma once

#include <d3d11.h>
#include "CudaDevice.h"
#include <wrl/client.h>
#include "../BinaryStream/BinaryReader.h"
#include "../BinaryStream/BinaryWriter.h"
#include "../ErrorLogger.h"

enum SolveMode
{
	STREAMLINE = 0,
	PATHLINE,
};

enum SeedingPattern
{
	SEED_RANDOM = 0,
	SEED_REGULAR,
	SEED_FILE,
};

enum IntegrationMethod
{
	EULER_METHOD = 0,
	MODIFIED_EULER,
	RK4_METHOD,
	RK5_METHOD,
};

enum InterpolationMethod
{
	DEFAULT_INTERPOLATION,
};

class CUDASolver
{
public:
	CUDASolver();

	bool Initialize(SolveMode _solveMode, SeedingPattern _seedingPattern, InterpolationMethod, unsigned int _InitialTimestep, unsigned _intFinalTimestep);
	bool Initialize(SolveMode _solveMode, SeedingPattern _seedingPattern, InterpolationMethod, unsigned int _InitialTimestep);
	bool Solve();

private:

	bool ReadField();
	bool SeedFiled();
	bool UploadField();
	bool SolveField(SolveMode mode);
	bool DrawField();
	bool DonwloadField();
	bool WriteField();


	CudaDevice cudaDevice;
	bool GetDevice();


	// Properties of the solver
	unsigned int m_initialTimestep = 0;
	unsigned int m_finalTimestep = 0;

	SolveMode m_mode;
	SeedingPattern m_seedingPattern;
	IntegrationMethod m_intergrationMehotd;
	InterpolationMethod m_interpolationMethod;



	// A COM pointer to the vector Field
	Microsoft::WRL::ComPtr<ID3D11Texture3D> m_resultTexture;
	Microsoft::WRL::ComPtr<ID3D11Device>  m_CudaDevice;
	
};