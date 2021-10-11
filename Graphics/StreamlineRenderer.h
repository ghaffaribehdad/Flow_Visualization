#pragma once

#include "LineRenderer.h"
#include "..//Cuda/StreamlineSolver.h"
#include "Vertex.h"

class StreamlineRenderer :public LineRenderer
{

public:

	ID3D11Texture2D* getOITTexture()
	{
		return OITTexture.Get();
	}

private:

	unsigned int n_occlusion = 50;
	StreamlineSolver streamlineSolver;

	Microsoft::WRL::ComPtr<ID3D11Buffer>				g_pStartOffsetBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer>				g_pFragmentAndLinkStructuredBuffer;

	Microsoft::WRL::ComPtr<ID3D11Texture2D>				OITTexture;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>		OITRenderTargetView;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>	OITResourceView;

	ID3D11UnorderedAccessView*	pUAV[2];
	ID3D11ShaderResourceView*	pSRV[2];

	ID3D11UnorderedAccessView*	g_pFragmentAndLinkStructuredBufferUAV = NULL;
	ID3D11UnorderedAccessView*	g_pStartOffsetBufferUAV = NULL;

	ID3D11ShaderResourceView*	g_pFragmentAndLinkStructuredBufferSRV = NULL;
	ID3D11ShaderResourceView*   g_pStartOffsetBufferSRV = NULL;

	ID3D11BlendState* g_pBlendStateNoBlend = NULL;
	ID3D11BlendState* g_pBlendStateSrcAlphaInvSrcAlphaBlend = NULL;
	ID3D11BlendState* g_pColorWritesOff = NULL;

	ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestDisabledStencilTestLessDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestEnabledStencilTestLessDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestDisabledDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestEnabledDSS = NULL;

	// Depth stencil buffer variables
	ID3D11Texture2D*            g_pDepthStencilTexture = NULL;
	ID3D11DepthStencilView*     g_pDepthStencilTextureDSV = NULL;

	Microsoft::WRL::ComPtr<ID3D11SamplerState>			samplerState;		// To sample the head of fragment in the second pass
	VertexBuffer<TexCoordVertex> vertexBufferQuadViewPlane;
	VertexBuffer<TexCoordVertex> vertexBufferQuadProjectionPlane;

	bool initializeOITBuffers();
	bool initializeOITViews();
	bool initializeSampler();
	bool initializeBlendandDepth();
	bool initializeQuadViewPlane();
	bool initializeQuadProjectionPlane();
	bool initializeOITRTV();
	void releaseResources();
	void updateBuffersSecondPass();
	void initializeResourcesOIT()
	{
		initializeRasterizer();
		initializeSampler();
		initializeOITBuffers();
		initializeOITViews();
		initializeBlendandDepth();
		initializeOITRTV();
		initializeQuadViewPlane();
		initializeQuadProjectionPlane();
	}


public:

	virtual void setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions, ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter * _adapter, const int & _width, const int & _height) override
	{
		this->solverOptions = &_solverOptions;
		this->solverOptions->p_Adapter = _adapter;
		this->renderingOptions = &_renderingOptions;
		this->device = _device;
		this->deviceContext = _deviceContext;
		this->width = _width;
		this->height = _height;

		releaseResources();
		initializeResourcesOIT();

	}



	virtual void show(RenderImGuiOptions* renderImGuiOptions) 
	{
		if (renderImGuiOptions->showStreamlines)
		{
			updateOIT = renderImGuiOptions->updateOIT;

			if (!streamlineSolver.checkFile(solverOptions))
			{
				ErrorLogger::Log("Cannot locate the file!");
				renderImGuiOptions->showStreamlines = false;
			}
			else
			{
				if (renderImGuiOptions->updateStreamlines)
				{
					this->updateScene();
					renderImGuiOptions->updateStreamlines = false;
				}
			}

		}

	}



	bool updateScene(bool WriteToFile = false)
	{
	
	
		this->vertexBuffer.reset();
		

		HRESULT hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer Streamlines.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();


		this->updateBuffers();
		return true;
	}




	void updateBuffers() override
	{
		
		if (solverOptions->loadNewfile && !solverOptions->updatePause)
		{


			if (solverOptions->fileLoaded)
			{
				this->streamlineSolver.releaseVolumeTexture();

			}
			this->streamlineSolver.Initialize(solverOptions);
			this->streamlineSolver.loadVolumeTexture();

			solverOptions->fileLoaded = true;
			solverOptions->loadNewfile = false;
		}
		else // do not load a new file
		{
			this->streamlineSolver.Reinitialize();
		}
		

		this->streamlineSolver.solve();
		this->streamlineSolver.FinalizeCUDA();
		//this->streamlineSolver.releaseVolumeIO();

		
	}


	bool setShaders_firstPass(D3D11_PRIMITIVE_TOPOLOGY Topology) 
	{
	
		pUAV[0] = g_pStartOffsetBufferUAV;
		pUAV[1] = g_pFragmentAndLinkStructuredBufferUAV;
		pSRV[0] = nullptr;
		pSRV[1] = nullptr;

		UINT initialCounts[2] = { 0,0 };
		UINT offset = 0;

		// Clear start offset buffer to -1
		const UINT dwClearDataMinusOne[1] = { 0xFFFFFFFF };
		this->deviceContext->ClearUnorderedAccessViewUint(g_pStartOffsetBufferUAV, dwClearDataMinusOne);
		this->deviceContext->ClearDepthStencilView(g_pDepthStencilTextureDSV,
			D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0, 0);

		this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());			// InputLayouut
		this->deviceContext->IASetPrimitiveTopology(Topology);								// Topology
		this->deviceContext->RSSetState(this->rasterizerstate.Get());						// Rasterizer
		this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);				// Vertex shader
		this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);				// Geometry shader
		this->deviceContext->PSSetShader(pixelshaderFirstPass.GetShader(), NULL, 0);		// Pixel shader
		this->deviceContext->OMSetBlendState(g_pColorWritesOff, 0, 0xffffffff);													// Blend state
		this->deviceContext->OMSetDepthStencilState(g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS, 0x00);			// Depth/Stencil
		this->deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(NULL, NULL, g_pDepthStencilTextureDSV, 1, 2, pUAV, initialCounts);	//Render Target and UAV 
		
		// The idea is to test before going to the pixel shader so the scene is correct
		//this->deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(NULL, NULL, depthstencil, 1, 2, pUAV, initialCounts);	
		//this->deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(NULL, NULL, depthstencil, 1, 2, pUAV, initialCounts);	
		return true;

	}

	bool setShaders_SecondPass(D3D11_PRIMITIVE_TOPOLOGY Topology)
	{

		UINT initialCounts[2] = { 0,0 };
		pUAV[0] = NULL;
		pUAV[1] = NULL;
		pSRV[0] = g_pStartOffsetBufferSRV;
		pSRV[1] = g_pFragmentAndLinkStructuredBufferSRV;

		//float rgb[4] = { 0,0,0,0 };
		//deviceContext->ClearRenderTargetView(OITRenderTargetView.Get(), rgb); 
		//deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(1, OITRenderTargetView.GetAddressOf(), NULL, 1, 2,pUAV, initialCounts);
		//deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(1, &mainRTV, depthstencil, 1, 2,pUAV, initialCounts);

		deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(1, &mainRTV, depthstencil, 1, 2,pUAV, initialCounts);
		this->deviceContext->PSSetShaderResources(0, 2, pSRV);
		this->deviceContext->RSSetState(this->rasterizerstate.Get());

		this->deviceContext->IASetInputLayout(this->vertexshaderSecondPass.GetInputLayout());				// Set the input layout
		this->deviceContext->IASetPrimitiveTopology(Topology);
		this->deviceContext->PSSetShader(pixelshaderSecondPass.GetShader(), NULL, 0);						// Pixel shader to blend
		this->deviceContext->VSSetShader(vertexshaderSecondPass.GetShader(), NULL, 0);						// Vertex shader to sample the texture and send it to pixel shader
		this->deviceContext->GSSetShader(NULL, NULL, NULL);													// Geometry shader is not needed
		UINT offset = 0;
		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBufferQuadViewPlane.GetAddressOf(), this->vertexBufferQuadViewPlane.StridePtr(), &offset);
		this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBufferSampler.GetAddressOf());
		this->deviceContext->OMSetBlendState(g_pBlendStateNoBlend, NULL, 0xFFFFFFFF);
		this->deviceContext->OMSetDepthStencilState(g_pDepthTestEnabledDSS, 0x00);							// Depth/Stencil
		//this->deviceContext->OMSetDepthStencilState(g_pDepthTestEnabledStencilTestLessDSS, 0x00);			// Depth/Stencil
		return true;
	}





	bool setShaders_ThirdPass(D3D11_PRIMITIVE_TOPOLOGY Topology)
	{
		UINT offset = 0;
		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBufferQuadProjectionPlane.GetAddressOf(), this->vertexBufferQuadProjectionPlane.StridePtr(), &offset);
		this->deviceContext->IASetInputLayout(this->vertexshaderSampler.GetInputLayout());		
		this->deviceContext->IASetPrimitiveTopology(Topology);

		this->deviceContext->VSSetConstantBuffers(0, 1, this->VS_SamplerConstantBuffer.GetAddressOf());
		this->deviceContext->VSSetShader(vertexshaderSampler.GetShader(), NULL, 0);

		this->deviceContext->RSSetState(this->rasterizerstate.Get());					

		this->deviceContext->PSSetShader(pixelShaderSampler.GetShader(), NULL, 0);				
		this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
		this->deviceContext->PSSetShaderResources(0, 1, OITResourceView.GetAddressOf());

		this->deviceContext->OMSetBlendState(g_pBlendStateNoBlend, NULL, 0xFFFFFFFF);
		this->deviceContext->OMGetRenderTargets(1, &mainRTV, &depthstencil);

		this->deviceContext->GSSetShader(NULL, NULL, NULL);								

		return true;
	}

	void Draw_firstPass()
	{
		switch (renderingOptions->drawMode)
		{
		case DrawMode::DrawMode::ADVECTION:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(solverOptions->currentSegment, i * solverOptions->lineLength);
			}

			solverOptions->currentSegment++;

			if (solverOptions->currentSegment == solverOptions->lineLength)
			{
				solverOptions->currentSegment = 0;
			}

			break;
		}

		case DrawMode::DrawMode::CURRENT:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(renderingOptions->lineLength, i * solverOptions->lineLength + (
					solverOptions->currentSegment - renderingOptions->lineLength + 1));
			}
			break;
		}
		case DrawMode::DrawMode::CURRENT_FULL:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(solverOptions->currentSegment, i * solverOptions->lineLength);
			}
			break;
		}
		case DrawMode::DrawMode::ADVECTION_FINAL:
		{
			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(renderingOptions->lineLength, i * solverOptions->lineLength + solverOptions->currentSegment);
			}

			solverOptions->currentSegment++;

			if (solverOptions->currentSegment == solverOptions->lineLength - renderingOptions->lineLength)
			{
				solverOptions->currentSegment = 0;
			}

			break;
		}

		default:
		{
			//this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count), 0);

			for (int i = 0; i < solverOptions->lines_count; i++)
			{
				this->deviceContext->Draw(llInt(solverOptions->lineLength), i * llInt(solverOptions->lineLength));

			}
			break;
		}


		}
	}



	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{


		if (solverOptions->usingTransparency)
		{
			if (solverOptions->viewChanged || updateOIT)
			{
				// First Pass
				setBuffers();
				updateConstantBuffer(camera);	// Parallel View
				setShaders_firstPass(Topology);	// No RT
				Draw_firstPass();				// Draw the streamlines and write them into startOffset and fragmenLinkedList buffers
				solverOptions->viewChanged = false;

			}
			// Second Pass
			updateBuffersSecondPass();
			setShaders_SecondPass(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);


			this->deviceContext->Draw(6, 0);

		}
		else
		{
			// First Pass
			initializeRasterizer();
			setShaders(Topology);
			updateConstantBuffer(camera);	
			setBuffers();
			solverOptions->viewChanged = false;

			

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
						solverOptions->currentIdx - solverOptions->firstIdx - renderingOptions->lineLength + 1));
				}
				break;
			}

			default:
			{
				this->deviceContext->Draw(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count), 0);
				break;
			}

			}
		}
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

		hr = this->VS_SamplerConstantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create vertex shader sampler constant buffer.");
			return false;
		}


		hr = this->PS_constantBufferSampler.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create pixel shader constant buffer.");
			return false;
		}
		return true;

	}


	void updateConstantBuffer(Camera& camera) override;

	virtual bool release() override
	{
		if (!this->releaseScene())
			return false;
		if (!this->streamlineSolver.release())
			return false;
		solverOptions->compressResourceInitialized = false;
		return true;

	}



	void updateConstantBufferSampler(Camera& camera)
	{


		DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

		// Set attributes of constant buffer for geometry shader
		VS_SamplerConstantBuffer.data.View = world * camera.GetViewMatrix();
		VS_SamplerConstantBuffer.data.Proj = camera.GetProjectionMatrix();
		// Update Constant Buffer
		VS_SamplerConstantBuffer.ApplyChanges();

	}
};