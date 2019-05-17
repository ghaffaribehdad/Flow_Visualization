
#include "CudaDevice.h"
#include "../ErrorLogger.h"
#include <vector>

// TODO: Too many include files
#include "../Graphics/AdapterReader.h"

bool CudaDevice::InitializeCUDA()
{
	int deviceCount = 0;
	int* pdeviceCount = &deviceCount;

	gpuErrchk(cudaGetDeviceCount(pdeviceCount));

	std::vector<AdapterData> adapters = AdapterReader::GetAdapters();


	gpuErrchk(cudaD3D11GetDevice(pdeviceCount, adapters[0].pAdapter));

	return true;
}

CudaDevice::CudaDevice()
{

}