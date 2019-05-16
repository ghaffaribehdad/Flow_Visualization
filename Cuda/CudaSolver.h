#pragma once

#include <d3d11.h>
#include "CudaDevice.h"
#include <wrl/client.h>

class CUDASolver
{
	
	bool Solve();

private:

	bool LoadField();
	bool ReleaseField();
	bool GetDevice();
	bool TransferField();
	bool CreateField();

	CudaDevice cudaDevice;

	// A COM pointer to the vector Field
	Microsoft::WRL::ComPtr<ID3D11Texture3D> Field;
	Microsoft::WRL::ComPtr<ID3D11Device>  device;
	
};