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

			switch (solverOptions->computationMode) {
			case(ComputationMode::ComputationMode::ONTHEFLY):
				if (!solverOptions->drawComplete) {
					this->updateSceneRealtime();
					renderImGuiOptions->updateRaycasting = true;
					renderImGuiOptions->updateFile[0] = true;
				}
				if (renderImGuiOptions->updateStreaklines) {
					this->resetRealtime();
					renderImGuiOptions->updateStreaklines = false;
				}
				break;

			case(ComputationMode::ComputationMode::PRECOMPUTATION):
				if (renderImGuiOptions->updateStreaklines) {
					this->updateScene();
					renderImGuiOptions->updateStreaklines = false;
				}
				break;
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
		this->streaklineSolver.Initialize(solverOptions, fieldOptions);
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
			this->streaklineSolver.ReinitializeCUDA();
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

		if (!this->streaklineSolver.initializeRealtime(solverOptions,fieldOptions)){
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
		GS_constantBuffer.data.gridDiameter.x = fieldOptions->gridDiameter[0];
		GS_constantBuffer.data.gridDiameter.y = fieldOptions->gridDiameter[1];
		GS_constantBuffer.data.gridDiameter.z = fieldOptions->gridDiameter[2];
		GS_constantBuffer.data.periodicity = solverOptions->periodic;
		if (solverOptions->projection == Projection::STREAK_PROJECTION)
		{
			GS_constantBuffer.data.particlePlanePos = streakProjectionPlane();
		}
		else if (solverOptions->projection == Projection::STREAK_PROJECTION_FIX)
		{
			GS_constantBuffer.data.particlePlanePos =  0;
		}



		GS_constantBuffer.data.streakPos = (float)solverOptions->projectPos * (solverOptions->gridDiameter[0] / (float)solverOptions->gridSize[0]);

		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;
		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.condition = solverOptions->usingTransparency;

		PS_constantBuffer.data.Ka = renderingOptions->Ka;
		PS_constantBuffer.data.Kd = renderingOptions->Kd;
		PS_constantBuffer.data.Ks = renderingOptions->Ks;
		PS_constantBuffer.data.shininessVal = renderingOptions->shininess;



		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}


};