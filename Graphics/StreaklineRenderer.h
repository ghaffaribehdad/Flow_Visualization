#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreaklineSolver.h"
#include "Vertex.h"

class StreaklineRenderer :public LineRenderer
{


private:

	StreaklineSolver streaklineSolver;
	int streakCounter = 0;

public:

	virtual bool release() override
	{


		if (!this->releaseScene())
			return false;
		if (!this->streaklineSolver.release())
			return false;
		streakCounter = 0;
		return true;


	}


	virtual void show(RenderImGuiOptions* renderImGuiOptions)
	{
		if (renderImGuiOptions->showStreaklines)
		{

			if (!streaklineSolver.checkFile(solverOptions))
			{
				ErrorLogger::Log("Cannot locate the file!");
				renderImGuiOptions->showStreaklines = false;
			}
			else
			{


				switch (solverOptions->computationMode)
				{
				case(ComputationMode::ComputationMode::ONTHEFLY):
				{

					if (!solverOptions->drawComplete)
					{
						this->updateSceneRealtime();
						renderImGuiOptions->updateRaycasting = true;
						renderImGuiOptions->fileChanged = true;
					}
					if (renderImGuiOptions->updateStreaklines)
					{
						this->resetRealtime();
						renderImGuiOptions->updateStreaklines = false;
					}
					break;
				}
				case(ComputationMode::ComputationMode::PRECOMPUTATION):
					if (renderImGuiOptions->updateStreaklines)
					{
						this->updateScene();
						renderImGuiOptions->updateStreaklines = false;
					}
					break;
				}

			}
			
			


		}
	}


	void resetRealtime() override
	{
		streakCounter = 0;
		counter = 0;
		this->streaklineSolver.resetRealtime();
	}

	bool updateScene()
	{
		this->vertexBuffer.Get()->Release();

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
		solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		this->streaklineSolver.Initialize(solverOptions);
		this->streaklineSolver.solve();
		this->streaklineSolver.FinalizeCUDA();

	}


	bool updateSceneRealtime()
	{
		if (streakCounter == 0)
		{
			this->initializeRealtime();
		}
		else
		{
			this->streaklineSolver.Reinitialize();
		}
		this->streaklineSolver.solveRealtime(streakCounter);
		this->streaklineSolver.FinalizeCUDA();
		this->solverOptions->currentIdx = solverOptions->firstIdx + streakCounter;
		return true;
	}



	bool initializeRealtime()
	{
		this->vertexBuffer.Get()->Release();
		HRESULT hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		if (!this->streaklineSolver.initializeRealtime(solverOptions))
		{
			return false;
		}

		return true;
	}




	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
		initializeRasterizer();
		setShaders(Topology);
		updateConstantBuffer(camera);
		setBuffers();


		switch (renderingOptions->drawMode)
		{


		case DrawMode::DrawMode::FULL:
		{
			this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count), 0);
			break;
		}

		case DrawMode::DrawMode::CURRENT:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(renderingOptions->lineLength, i * solverOptions->lineLength + (
					solverOptions->currentIdx - solverOptions->firstIdx - renderingOptions->lineLength));
			}
			break;
		}

		default:
		{
			this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count), 0);
			break;
		}

		}
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

		GS_constantBuffer.data.View = world * camera.GetViewMatrix();
		GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
		GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
		GS_constantBuffer.data.tubeRadius = renderingOptions->tubeRadius;
		GS_constantBuffer.data.viewDir = camera.GetViewVector();
		GS_constantBuffer.data.projection = solverOptions->projection;
		GS_constantBuffer.data.gridDiameter.x = solverOptions->gridDiameter[0];
		GS_constantBuffer.data.gridDiameter.y = solverOptions->gridDiameter[1];
		GS_constantBuffer.data.gridDiameter.z = solverOptions->gridDiameter[2];
		GS_constantBuffer.data.periodicity = solverOptions->periodic;
		GS_constantBuffer.data.particlePlanePos = streakProjectionPlane();
		GS_constantBuffer.data.streakPos = (float)solverOptions->projectPos * (solverOptions->gridDiameter[0] / (float)solverOptions->gridSize[0]);

		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;
		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.condition = solverOptions->usingTransparency;


		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}


};