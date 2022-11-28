#pragma once

#include "LineRenderer.h"
#include "..//Cuda/PathlineSolver.h"
#include "Vertex.h"

class PathlineRenderer :public LineRenderer
{


private:

	PathlineSolver pathlinesolver;
	int pathCounter = 0;

public:

	virtual bool release() override
	{
		if (!this->releaseScene())
			return false;
		if (!this->pathlinesolver.release())
			return false;
		pathCounter = 0;
		return true;
	}

	virtual void show(RenderImGuiOptions* renderImGuiOptions) 
	{
		if (renderImGuiOptions->showPathlines)
		{

			if (!pathlinesolver.checkFile())
			{
				ErrorLogger::Log("Cannot locate the file!");
				renderImGuiOptions->showPathlines = false;
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
						renderImGuiOptions->updatefluctuation = true;
					}

					if (renderImGuiOptions->updatePathlines)
					{
						this->resetRealtime();
						renderImGuiOptions->updatePathlines = false;
					}
					break;

				}
				case(ComputationMode::ComputationMode::PRECOMPUTATION):
				{

					if (renderImGuiOptions->updatePathlines)
					{
						this->updateScene();
						renderImGuiOptions->updatePathlines = false;
					}

					break;
				}

				}
			}


			
		}

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

		if (!this->pathlinesolver.initializeRealtime(solverOptions))
		{
			return false;
		}

		return true;
	}


	bool updateSceneRealtime()
	{
		if (pathCounter == 0)
		{
			this->initializeRealtime();
		}
		else
		{
			this->pathlinesolver.ReinitializeCUDA();
		}
		this->pathlinesolver.solveRealtime(pathCounter);
		this->pathlinesolver.FinalizeCUDA();
		this->solverOptions->currentIdx = solverOptions->firstIdx + pathCounter;
		return true;
	}


	void resetRealtime() override
	{

		pathCounter = 0;
		this->counter = 0;
		this->pathlinesolver.resetRealtime();

	}

	void updateBuffers() override
	{

		this->pathlinesolver.Initialize(solverOptions,fieldOptions);
		this->pathlinesolver.solve();
		this->pathlinesolver.FinalizeCUDA();

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
					solverOptions->currentIdx - solverOptions->firstIdx - renderingOptions->lineLength+1));
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
		GS_constantBuffer.data.particlePlanePos = streakProjectionPlane();
		GS_constantBuffer.data.transparencyMode = solverOptions->transparencyMode;
		GS_constantBuffer.data.timDim = fieldOptions->lastIdx - fieldOptions->firstIdx + 1 ;
		GS_constantBuffer.data.currentTime = solverOptions->currentIdx - fieldOptions->firstIdx;
		GS_constantBuffer.data.usingThreshold = solverOptions->usingThreshold;
		GS_constantBuffer.data.threshold = solverOptions->transparencyThreshold;



		GS_constantBuffer.data.streakPos = (float)solverOptions->projectPos * (solverOptions->gridDiameter[0] /(float) solverOptions->gridSize[0]);

		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;
		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.condition = solverOptions->usingTransparency;

		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;
		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.viewportWidth = width;
		PS_constantBuffer.data.viewportHeight = height;
		PS_constantBuffer.data.Ka = renderingOptions->Ka;
		PS_constantBuffer.data.Kd = renderingOptions->Kd;
		PS_constantBuffer.data.Ks = renderingOptions->Ks;
		PS_constantBuffer.data.shininessVal = renderingOptions->shininess;


		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}


};