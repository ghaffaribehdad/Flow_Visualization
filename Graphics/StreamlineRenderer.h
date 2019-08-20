#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.cuh"
#include "Vertex.h"

class StreamlineRenderer :public LineRenderer
{
private:

	StreamlineSolver<float> streamlineSolver;

	//Vertex vertex[3] = { {0,0,0,0,1,0,0,1},{0,1,0,0,1,0,0,1},{0,2,0,0,1,0,0,1} };


	void updateVertexBuffer()
	{
		
		solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		this->streamlineSolver.Initialize(*solverOptions);
		this->streamlineSolver.solve();
		this->streamlineSolver.FinalizeCUDA();		
	}


public:
	void updateScene() override
	{
		this->updateVertexBuffer();
		this->updateIndexBuffer();
	}
		
};