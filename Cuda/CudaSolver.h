#pragma once

#include <d3d11.h>
#include "CudaDevice.h"
#include <wrl/client.h>
#include "../BinaryStream/BinaryReader.h"
#include "../BinaryStream/BinaryWriter.h"
#include "../ErrorLogger.h"
#include <DirectXMath.h>

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

	bool Initialize
	(
		SolveMode _solveMode, 
		SeedingPattern _seedingPattern,
		IntegrationMethod _integrationMethod,
		InterpolationMethod _interpolationMethod,
		unsigned int _InitialTimestep,
		unsigned _intFinalTimestep
	);

	bool Solve();

protected:

	bool ReadField(std::vector<char>* p_vec_buffer, std::string fileName);
	bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox);
	bool UploadField();
	bool SolveField(SolveMode mode);
	bool DrawField();
	bool DonwloadField();
	bool WriteField();

	CudaDevice cudaDevice;
	bool GetDevice();

	// Input of Solver
	DirectX::XMFLOAT3 * seedingBox;

	SolveMode m_solveMode;
	SeedingPattern m_seedingPattern;
	IntegrationMethod m_intergrationMehotd;
	InterpolationMethod m_interpolationMethod;
	unsigned int m_initialTimestep;
	unsigned int m_finalTimeStep;


	// A COM pointer to the vector Field
	Microsoft::WRL::ComPtr<ID3D11Texture3D> m_resultTexture;
	Microsoft::WRL::ComPtr<ID3D11Device>  m_CudaDevice;
	
};