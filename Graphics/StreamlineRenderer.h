#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.h"
#include "Vertex.h"

class StreamlineRenderer :public LineRenderer
{


private:
	StreamlineSolver streamlineSolver;

public:

	virtual void show(RenderImGuiOptions* renderImGuiOptions) 
	{
		if (renderImGuiOptions->showStreamlines)
		{
			if (renderImGuiOptions->updateStreamlines && renderImGuiOptions->streamlineGenerating)
			{
				this->updateScene(true);
				if (solverOptions->counter < solverOptions->fileToSave)
				{
					solverOptions->counter++;
				}
				else
				{
					renderImGuiOptions->updateStreamlines = false;
				}
			}
			else if (renderImGuiOptions->updateStreamlines)
			{
				this->updateScene();
				renderImGuiOptions->updateStreamlines = false;

			}
		}
	}

	bool updateScene(bool WriteToFile = false)
	{

		// IT HAS MOMORY LEAK WHY WHY ???
		this->vertexBuffer.Get()->Release();
		HRESULT hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		//if (WriteToFile)
		//{
		//	this->updateBuffersAndWriteToFile();
		//}
		//else
		//{
		//	this->updateBuffers();
		//}

		this->updateBuffers();
		return true;
	}

	void updateBuffers() override
	{
		
		this->streamlineSolver.Initialize(solverOptions);
		this->streamlineSolver.solve();
		this->streamlineSolver.FinalizeCUDA();
		
	}


	//void updateBuffersAndWriteToFile() 
	//{

	//	this->streamlineSolver.Initialize(solverOptions);
	//	this->streamlineSolver.solveAndWrite();
	//	this->streamlineSolver.FinalizeCUDA();

	//}

	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{


		initializeRasterizer();
		setShaders(Topology);
		updateConstantBuffer(camera);
		setBuffers();


		


		switch (renderingOptions->drawMode)
		{
		case DrawMode::DrawMode::STATIONARY:
		{
			this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count),0);
			break;
		}
		case DrawMode::DrawMode::ADVECTION:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(counter, i * solverOptions->lineLength);

			}

			this->counter++;

			if (counter == solverOptions->lineLength)
			{
				counter = 0;
			}

			break;
		}

		case DrawMode::DrawMode::CURRENT:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(renderingOptions->lineLength, i * solverOptions->lineLength + counter);
			}

			this->counter++;

			if (counter == solverOptions->lineLength- renderingOptions->lineLength)
			{
				counter = 0;
			}

			break;
		}
		}

		this->cleanPipeline();
	}

	//bool flushVerteBuffer()
	//{

	//	D3D11_MAPPED_SUBRESOURCE mappedResource;
	//	ZeroMemory(&mappedResource, sizeof(D3D11_MAPPED_SUBRESOURCE));

	//	HRESULT hr;
	//	hr = this->deviceContext->Map(vertexBuffer.Get(), NULL, D3D11_MAP_READ_WRITE, NULL, &mappedResource);
	//	if (FAILED(hr))
	//	{
	//		ErrorLogger::Log(hr, "Failed to Map Vertex Buffer");
	//		return false;
	//	}

	//	VertexBuffer<Vertex>* p_buffer = new VertexBuffer<Vertex>[solverOptions->lineLength * solverOptions->lines_count];
	//	memcpy(p_buffer, mappedResource.pData, sizeof(solverOptions->lineLength * solverOptions->lines_count * sizeof(VertexBuffer<Vertex>)));

	//	this->deviceContext->Unmap(vertexBuffer.Get(), NULL);

	//	delete[] p_buffer;

	//}
		

	bool initializeBuffers() override
	{

		HRESULT hr = this->GS_constantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Geometry Shader Constant buffer.");
			return false;
		}

		hr = this->PS_constantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Pixel shader Constant buffer.");
			return false;
		}



		//Dummy Vertex Buffer which will be expand to the desired size in UpdateScene
		hr = this->vertexBuffer.Initialize(this->device, NULL, 1);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		return true;

	}


	void updateConstantBuffer(Camera& camera) override
	{


		DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

		// Set attributes of constant buffer for geometry shader
		GS_constantBuffer.data.View = world * camera.GetViewMatrix();
		GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
		GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
		GS_constantBuffer.data.tubeRadius = renderingOptions->tubeRadius;
		GS_constantBuffer.data.viewDir = camera.GetViewVector();


		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;

		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.isRaycasting = renderingOptions->isRaycasting;


		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}


};