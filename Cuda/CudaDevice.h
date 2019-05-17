#pragma once

class CudaDevice
{

public:

	CudaDevice();

	const float & GetMemoryPower() const;
	const float& GetComputePower() const;
	bool InitializeCUDA();

private:

	float computePower;
	float memoryPower;


};