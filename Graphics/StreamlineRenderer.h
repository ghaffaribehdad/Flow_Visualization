#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.cuh"
#include "Vertex.h"

class StreamlineRenderer :public LineRenderer
{
public:

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
	void updateBuffers() override
	{
		this->updateIndexBuffer();
		this->updateVertexBuffer();
		
	}
		
};