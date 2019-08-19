#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.cuh"

class StreamlineRenderer :protected LineRenderer
{
private:

	StreamlineSolver<float> streamlineSolver;



	void updateVertexBuffer()
	{
		this->streamlineSolver.Initialize(this->solverOptions);
		this->streamlineSolver.solve();
		this->streamlineSolver.FinalizeCUDA();

		this->solverOptions.beginStream = false;
		
	}





public:
	void update(Camera& camera) override
	{
		this->updateConstantBuffer(camera);
		this->updateVertexBuffer();
		this->updateIndexBuffer();
	}
		
};