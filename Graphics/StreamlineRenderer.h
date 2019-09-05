#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.h"
#include "Vertex.h"

class StreamlineRenderer :public LineRenderer
{


private:

	StreamlineSolver streamlineSolver;

public:

	bool updateScene()
	{

		HRESULT hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		this->updateBuffers();

		return true;

	}

	void updateBuffers() override
	{
		
		this->streamlineSolver.Initialize(solverOptions);
		this->streamlineSolver.solve();
		this->streamlineSolver.FinalizeCUDA();
		
	}

	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
		initilizeRasterizer();
		setShaders(Topology);
		updateConstantBuffer(camera);
		setBuffers();
		this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count),0);
		this->cleanPipeline();
	}
		

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



		hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
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