#include "CudaSolver.h"

template class CUDASolver<float>;
template class CUDASolver<double>;

template <typename T>
CUDASolver<T>::CUDASolver()
{
	std::printf("A solver is created!\n");
}

// Initilize the solver
template <typename T>
bool CUDASolver<T>::Initialize(SolverOptions _solverOptions)
{
	this->solverOptions = _solverOptions;
	this->InitializeCUDA();
	
	return true;
}




bool SeedFiled(SeedingPattern, DirectX::XMFLOAT3 dimenions, DirectX::XMFLOAT3 seedbox)
{
	return true;
}

template <typename T>
bool CUDASolver<T>::FinalizeCUDA()
{
	gpuErrchk(cudaGraphicsUnmapResources(1,	&this->cudaGraphics	));
	return true;
}

template <typename T>
bool CUDASolver<T>::InitializeCUDA()
{
	// Get number of CUDA-Enable devices
	int device;
	gpuErrchk(cudaD3D11GetDevice(&device,solverOptions.p_Adapter));

	// Get properties of the Best(usually at slot 0) card
	gpuErrchk(cudaGetDeviceProperties(&this->cuda_device_prop, 0));

	// Register Vertex Buffer to map it
	gpuErrchk(cudaGraphicsD3D11RegisterResource(
		&this->cudaGraphics,
		this->solverOptions.p_vertexBuffer,
		cudaGraphicsRegisterFlagsNone));

	// Map Vertex Buffer
	gpuErrchk(cudaGraphicsMapResources(
		1,
		&this->cudaGraphics
		));

	// Get Mapped pointer (why 3?)
	size_t size = 3*std::size_t(sizeof(Vertex));

	gpuErrchk(cudaGraphicsResourceGetMappedPointer(
		&p_VertexBuffer,
		&size,
		this->cudaGraphics
	));

	return true;
}


template <typename T>
bool CUDASolver<T>::InitializeVolumeIO()
{
	this->volume_IO.setFileName(this->solverOptions.fileName);
	this->volume_IO.setFilePath(this->solverOptions.filePath);
	this->volume_IO.setIndex(this->solverOptions.firstIdx, this->solverOptions.lastIdx);
	return true;
}