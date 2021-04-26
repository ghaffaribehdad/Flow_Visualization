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

	ID3D11BlendState* g_pBlendStateNoBlend;
	ID3D11BlendState* g_pBlendStateSrcAlphaInvSrcAlphaBlend;
	ID3D11BlendState* g_pColorWritesOff;

	ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestDisabledDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestEnabledNoDepthWritesDSS = NULL;
	ID3D11DepthStencilState*    g_pDepthTestEnabledDSS = NULL;

	// Depth stencil buffer variables
	ID3D11Texture2D*            g_pDepthStencilTexture = NULL;
	ID3D11DepthStencilView*     g_pDepthStencilTextureDSV = NULL;

	Microsoft::WRL::ComPtr<ID3D11SamplerState>			samplerState;		// To sample the head of fragment in the second pass
	VertexBuffer<TexCoordVertex> vertexBufferQuad;


	bool initializeQuad()
	{
		TexCoordVertex quad[] =
		{
				TexCoordVertex(-1.0f,	-1.0f,	1.0f,	0.0f,	1.0f), //Bottom Left 
				TexCoordVertex(-1.0f,	1.0f,	1.0f,	0.0f,	0.0f), //Top Left
				TexCoordVertex(1.0f,	1.0f,	1.0f,	1.0f,	0.0f), //Top Right

				TexCoordVertex(-1.0f,	-1.0f,	1.0f,	0.0f,	1.0f), //Bottom Left 
				TexCoordVertex(1.0f,	1.0f,	1.0f,	1.0f,	0.0f), //Top Right
				TexCoordVertex(1.0f,	-1.0f,	1.0f,	1.0f,	1.0f), //Bottom Right

		};

		HRESULT hr = this->vertexBufferQuad.Initialize(this->device, quad, ARRAYSIZE(quad));
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create offset buffer.");
			return false;
		}


		return true;
	}

	bool initializeProjectionPlane()
	{

		TexCoordVertex quad[] =
		{
			TexCoordVertex(0,	-solverOptions->gridDiameter[1] / 2.0f,	-solverOptions->gridDiameter[2] / 2.0f,	0.0f,	1.0f), //Bottom Left 
			TexCoordVertex(0,	 solverOptions->gridDiameter[1] / 2.0f,	-solverOptions->gridDiameter[2] / 2.0f,	0.0f,	0.0f), //Top Left
			TexCoordVertex(0,	 solverOptions->gridDiameter[1] / 2.0f,	 solverOptions->gridDiameter[2] / 2.0f,	1.0f,	0.0f), //Top Right

			TexCoordVertex(0,	-solverOptions->gridDiameter[1] / 2.0f,	-solverOptions->gridDiameter[2] / 2.0f,	0.0f,	1.0f), //Bottom Left 
			TexCoordVertex(0,	 solverOptions->gridDiameter[1] / 2.0f,	 solverOptions->gridDiameter[2] / 2.0f,	1.0f,	0.0f), //Top Right
			TexCoordVertex(0,	-solverOptions->gridDiameter[1] / 2.0f,	 solverOptions->gridDiameter[2] / 2.0f,	1.0f,	1.0f), //Bottom Right

		};

	

		HRESULT hr = this->vertexBufferQuad.Initialize(this->device, quad, ARRAYSIZE(quad));
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create offset buffer.");
			return false;
		}

		return true;
	}





	bool firstPassInitialization()
	{
		HRESULT hr;	


		// Create Start Offset buffer
		D3D11_BUFFER_DESC OffsetBufferDesc;
		OffsetBufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		OffsetBufferDesc.ByteWidth = width * height * sizeof(UINT);
		OffsetBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
		OffsetBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		OffsetBufferDesc.CPUAccessFlags = 0;
		OffsetBufferDesc.StructureByteStride = 0;
		hr = device->CreateBuffer(&OffsetBufferDesc, NULL, g_pStartOffsetBuffer.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create offset buffer.");
			return false;
		}



		// Create UAV view of Start Offset buffer
		D3D11_UNORDERED_ACCESS_VIEW_DESC UAVOffsetBufferDesc;
		UAVOffsetBufferDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		UAVOffsetBufferDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		UAVOffsetBufferDesc.Buffer.FirstElement = 0;
		UAVOffsetBufferDesc.Buffer.NumElements = width * height;
		UAVOffsetBufferDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
		hr = device->CreateUnorderedAccessView(g_pStartOffsetBuffer.Get(), &UAVOffsetBufferDesc, &g_pStartOffsetBufferUAV);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create offset buffer UAV View.");
			return false;
		}

		// Create SRV view of Start Offset buffer
		D3D11_SHADER_RESOURCE_VIEW_DESC SRVOffsetBufferDesc;
		SRVOffsetBufferDesc.Format = DXGI_FORMAT_R32_UINT;
		SRVOffsetBufferDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		SRVOffsetBufferDesc.Buffer.ElementOffset = 0;
		SRVOffsetBufferDesc.Buffer.ElementWidth = (DWORD)(width * height);
		hr = device->CreateShaderResourceView(g_pStartOffsetBuffer.Get(), &SRVOffsetBufferDesc, &g_pStartOffsetBufferSRV);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create offset buffer SRV View.");
			return false;
		}

		D3D11_BUFFER_DESC BufferDesc;
		BufferDesc.CPUAccessFlags = 0;
		BufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		BufferDesc.ByteWidth = (DWORD)( 100 * width * height *
			sizeof(LinkedList));
		BufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		BufferDesc.Usage = D3D11_USAGE_DEFAULT;
		BufferDesc.StructureByteStride = sizeof(LinkedList);
		hr = device->CreateBuffer(&BufferDesc, NULL, g_pFragmentAndLinkStructuredBuffer.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create UAV link list buffer!");
			return false;
		}

		// Create UAV view of Fragment and Link Buffer
		D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
		UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.FirstElement = 0;
		UAVDesc.Buffer.NumElements = (DWORD)(100 * width * height);
		UAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;
		hr = device->CreateUnorderedAccessView(g_pFragmentAndLinkStructuredBuffer.Get(), &UAVDesc,
			&g_pFragmentAndLinkStructuredBufferUAV);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create UAV link list buffer view!");
			return false;
		}


		D3D11_SHADER_RESOURCE_VIEW_DESC SRVBufferDesc;
		// Create SRV view of Fragment and Link Buffer
		SRVBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
		SRVBufferDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		SRVBufferDesc.Buffer.ElementOffset = 0;
		SRVBufferDesc.Buffer.ElementWidth = (DWORD)(100 * width * height);
		hr = device->CreateShaderResourceView(g_pFragmentAndLinkStructuredBuffer.Get(), &SRVBufferDesc,
			&g_pFragmentAndLinkStructuredBufferSRV);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create SRV link list buffer view!");
			return false;
		}


		//Create sampler description for sampler state
		D3D11_SAMPLER_DESC sampDesc;
		ZeroMemory(&sampDesc, sizeof(sampDesc));
		sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		sampDesc.MinLOD = 0;
		sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
		 hr = this->device->CreateSamplerState(&sampDesc, this->samplerState.GetAddressOf()); //Create sampler state
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to create sampler state.");
			return false;
		}

		pUAV[0] = g_pStartOffsetBufferUAV;
		pUAV[1] = g_pFragmentAndLinkStructuredBufferUAV;
		pSRV[0] = g_pStartOffsetBufferSRV;
		pSRV[1] = g_pFragmentAndLinkStructuredBufferSRV;

		initializeQuad();


		// Create a blend state to disable alpha blending
		D3D11_BLEND_DESC BlendState;
		ZeroMemory(&BlendState, sizeof(D3D11_BLEND_DESC));
		BlendState.IndependentBlendEnable = FALSE;
		BlendState.RenderTarget[0].BlendEnable = FALSE;
		BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		hr = device->CreateBlendState(&BlendState, &g_pBlendStateNoBlend);

		// Create a blend state to enable alpha blending
		ZeroMemory(&BlendState, sizeof(D3D11_BLEND_DESC));
		BlendState.IndependentBlendEnable = TRUE;
		BlendState.RenderTarget[0].BlendEnable = TRUE;
		BlendState.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
		BlendState.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
		BlendState.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
		BlendState.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		BlendState.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
		BlendState.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
		BlendState.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		hr = device->CreateBlendState(&BlendState, &g_pBlendStateSrcAlphaInvSrcAlphaBlend);


		// Create a blend state to disable color writes
		BlendState.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
		BlendState.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
		BlendState.RenderTarget[0].RenderTargetWriteMask = 0;
		hr = device->CreateBlendState(&BlendState, &g_pColorWritesOff);





		D3D11_DEPTH_STENCIL_DESC DSDesc;
		DSDesc.DepthEnable = FALSE;
		DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
		DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
		DSDesc.StencilEnable = FALSE;
		DSDesc.StencilReadMask = 0xff;
		DSDesc.StencilWriteMask = 0xff;
		DSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		DSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestDisabledDSS);
		DSDesc.DepthEnable = TRUE;
		DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
		DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
		DSDesc.StencilEnable = FALSE;
		hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledNoDepthWritesDSS);
		DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
		DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledDSS);
		
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create depth stencil!");
			return false;
		}

			
		DSDesc.DepthEnable = TRUE;
		DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
		DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
		DSDesc.StencilEnable = TRUE;
		DSDesc.StencilReadMask = 0xFF;
		DSDesc.StencilWriteMask = 0xFF;
		DSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_INCR_SAT;
		DSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		DSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
		DSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_INCR_SAT;
		DSDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
		hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create depth stencil!");
			return false;
		}

		// Create Depth Stencil View
		D3D11_DEPTH_STENCIL_VIEW_DESC SRVDepthStencilDesc;
		SRVDepthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		SRVDepthStencilDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		SRVDepthStencilDesc.Texture2D.MipSlice = 0;
		SRVDepthStencilDesc.Flags = 0;
		hr = device->CreateDepthStencilView(g_pDepthStencilTexture, &SRVDepthStencilDesc,
			&g_pDepthStencilTextureDSV);

		return true;
	}


public:


	virtual bool initializeShaders() override
	{
		LineRenderer::initializeShaders();
		firstPassInitialization();

		return true;

	}


	virtual void show(RenderImGuiOptions* renderImGuiOptions) 
	{
		if (renderImGuiOptions->showStreamlines)
		{

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
	
		if (vertexBuffer.initialized())
		{
			this->vertexBuffer.reset();
		}

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
		
		if (solverOptions->loadNewfile && !solverOptions->updatePause)
		{


			if (solverOptions->fileLoaded)
			{
				this->streamlineSolver.releaseVolumeIO();
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
		this->streamlineSolver.releaseVolumeIO();

		
	}


	virtual bool setShaders(D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
	
		UINT initialCounts[2] = { 0,0 };
		UINT offset = 0;

		// Clear start offset buffer to -1
		const UINT dwClearDataMinusOne[1] = { 0xFFFFFFFF };
		this->deviceContext->ClearUnorderedAccessViewUint(g_pStartOffsetBufferUAV, dwClearDataMinusOne);

		this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());			// InputLayouut
		this->deviceContext->IASetPrimitiveTopology(Topology);								// Topology
		this->deviceContext->RSSetState(this->rasterizerstate.Get());						// Rasterizer
		this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);				// Vertex shader
		this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);				// Geometry shader
		this->deviceContext->PSSetShader(pixelshaderFirstPass.GetShader(), NULL, 0);		// Pixel shader
		this->deviceContext->OMSetBlendState(g_pColorWritesOff, 0, 0xffffffff);													// Blend state
		//this->deviceContext->OMSetBlendState(g_pBlendStateNoBlend, 0, 0xffffffff);													// Blend state
		this->deviceContext->OMSetDepthStencilState(g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS, 0x00);			// Depth/Stencil
		this->deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(NULL, NULL, depthstencil, 1, 2, pUAV, initialCounts);	//Render Target and UAV 
		return true;

	}

	bool setShaders_SecondPass(D3D11_PRIMITIVE_TOPOLOGY Topology)
	{

		UINT initialCounts[2] = { 0,0 };
		pUAV[0] = NULL;
		pUAV[1] = NULL;

		deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(1, OITRenderTargetView.GetAddressOf(), NULL, 1, 2,pUAV, initialCounts);
		this->deviceContext->PSSetShaderResources(0, 2, pSRV);
		this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
		this->deviceContext->IASetInputLayout(this->vertexshaderSecondPass.GetInputLayout());		// Set the input layout
		this->deviceContext->IASetPrimitiveTopology(Topology);
		this->deviceContext->PSSetShader(pixelshaderSecondPass.GetShader(), NULL, 0);				// Pixel shader to blend
		this->deviceContext->VSSetShader(vertexshaderSecondPass.GetShader(), NULL, 0);	// Vertex shader to sample the texture and send it to pixel shader
		this->deviceContext->GSSetShader(NULL, NULL, NULL);								// Geometry shader is not needed
		UINT offset = 0;
		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBufferQuad.GetAddressOf(), this->vertexBufferQuad.StridePtr(), &offset);
		this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBuffer.GetAddressOf());
		this->deviceContext->OMSetBlendState(g_pBlendStateNoBlend, NULL, 0xFFFFFFFF);
		this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
		return true;
	}

	bool setShaders_ThirdPass(D3D11_PRIMITIVE_TOPOLOGY Topology)
	{
		UINT offset = 0;
		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBufferQuad.GetAddressOf(), this->vertexBufferQuad.StridePtr(), &offset);
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

	void Draw_secondPass()
	{
		this->deviceContext->Draw(6, 0);
	}

	void cleanPipeline()
	{

		pUAV[0] = nullptr;
		pUAV[1] = nullptr;

		pSRV[0] = nullptr;
		pSRV[1] = nullptr;

		SAFE_RELEASE(g_pStartOffsetBufferUAV);
		SAFE_RELEASE(g_pStartOffsetBufferSRV);

		SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferSRV);
		SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferUAV);

		SAFE_RELEASE(g_pBlendStateSrcAlphaInvSrcAlphaBlend);
		SAFE_RELEASE(g_pColorWritesOff);

		SAFE_RELEASE(g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS);
		SAFE_RELEASE(g_pDepthTestDisabledDSS);
		SAFE_RELEASE(g_pDepthTestEnabledNoDepthWritesDSS);
		SAFE_RELEASE(g_pDepthTestEnabledDSS);
		SAFE_RELEASE(g_pDepthStencilTextureDSV);
		SAFE_RELEASE(g_pDepthStencilTexture);



		g_pFragmentAndLinkStructuredBuffer.Reset();
		g_pStartOffsetBuffer.Reset();
		blendState.Reset();
		OITTexture.Reset();
		OITRenderTargetView.Reset();
		vertexBufferQuad.reset();
	}

	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
		if (solverOptions->viewChanged)
		{
			cleanPipeline();
			firstPassInitialization();
			initializeRasterizer();

			// First Pass
			setBuffers();
			updateConstantBuffer(camera);
			setShaders(Topology);
			Draw_firstPass();

			// Second Pass
			initializeOITRTV();
			setShaders_SecondPass(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			this->deviceContext->Draw(6, 0);
			vertexBufferQuad.reset();

			solverOptions->viewChanged = false;
		}

		// Third pass
		OITRenderTargetView.Reset();			// Release OIT RTV
		initializeRasterizer();
		initializeProjectionPlane();	//Initialize vertex buffer
		initializeThirdPassResources();
		updateConstantBufferSampler(camera);
		setShaders_ThirdPass(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

		this->deviceContext->Draw(6, 0);

		cleanPipelineSampler();
	}


	void cleanPipelineSampler()
	{
		samplerState.Reset();
		vertexBufferQuad.reset();
		rasterizerstate.Reset();
		OITRenderTargetView.Reset();
		OITResourceView.Reset();
		SAFE_RELEASE(g_pBlendStateNoBlend);

	}
	bool initializeThirdPassResources()
	{
		HRESULT hr;

		//Create sampler description for sampler state
		D3D11_SAMPLER_DESC sampDesc;
		ZeroMemory(&sampDesc, sizeof(sampDesc));
		sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
		sampDesc.MinLOD = 0;
		sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
		hr = this->device->CreateSamplerState(&sampDesc, this->samplerState.GetAddressOf()); //Create sampler state
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to create sampler state.");
			return false;
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC shader_resource_view_desc;
		ZeroMemory(&shader_resource_view_desc, sizeof(shader_resource_view_desc));
		shader_resource_view_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		shader_resource_view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;

		hr = this->device->CreateShaderResourceView(
			OITTexture.Get(),
			&shader_resource_view_desc,
			OITResourceView.GetAddressOf()
		);

		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create shader resource view");
			return false;
		}

		return true;
	}
		
	void initializeOITRTV()
	{
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));

	textureDesc.ArraySize = 1;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET |D3D11_BIND_SHADER_RESOURCE ;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.Height = height;
	textureDesc.Width = width;
	textureDesc.MipLevels = 1;
	textureDesc.MiscFlags = 0;
	textureDesc.SampleDesc.Count = 4;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;


	HRESULT hr = this->device->CreateTexture2D(&textureDesc, nullptr, this->OITTexture.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Front Texture");
	}

	// Create Render targe view
	hr = this->device->CreateRenderTargetView(OITTexture.Get(), NULL, this->OITRenderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create RenderTargetView");
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

		return true;

	}


	void updateConstantBuffer(Camera& camera) override
	{
		DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();
		//GS_constantBuffer.data.View = world * camera.GetViewMatrix();
		GS_constantBuffer.data.View = world * camera.GetViewMatrix(DirectX::XMFLOAT3(0, 0, 0));
		GS_constantBuffer.data.Proj = camera.GetParallelProjectionMatrix();
		//GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
		GS_constantBuffer.data.eyePos = DirectX::XMFLOAT3(0, 0, 0);
		GS_constantBuffer.data.tubeRadius = renderingOptions->tubeRadius;
		//GS_constantBuffer.data.viewDir = camera.GetViewVector();
		GS_constantBuffer.data.viewDir = DirectX::XMFLOAT3(0, 0 , 1);
		GS_constantBuffer.data.projection = solverOptions->projection;
		GS_constantBuffer.data.gridDiameter.x = solverOptions->gridDiameter[0];
		GS_constantBuffer.data.gridDiameter.y = solverOptions->gridDiameter[1];
		GS_constantBuffer.data.gridDiameter.z = solverOptions->gridDiameter[2];
		GS_constantBuffer.data.periodicity = solverOptions->periodic;
		GS_constantBuffer.data.particlePlanePos = streakProjectionPlane();
		GS_constantBuffer.data.streakPos = solverOptions->projectPos * (solverOptions->gridDiameter[0] / solverOptions->gridSize[0]);
		GS_constantBuffer.data.transparencyMode = solverOptions->transparencyMode;
		GS_constantBuffer.data.timDim = solverOptions->lineLength;
		GS_constantBuffer.data.currentTime = solverOptions->firstIdx - solverOptions->currentIdx;
		GS_constantBuffer.data.usingThreshold = solverOptions->usingThreshold;
		GS_constantBuffer.data.threshold = solverOptions->transparencyThreshold;

		PS_constantBuffer.data.minMeasure = renderingOptions->minMeasure;
		PS_constantBuffer.data.maxMeasure = renderingOptions->maxMeasure;
		PS_constantBuffer.data.minColor = DirectX::XMFLOAT4(renderingOptions->minColor);
		PS_constantBuffer.data.maxColor = DirectX::XMFLOAT4(renderingOptions->maxColor);
		PS_constantBuffer.data.condition = solverOptions->usingTransparency;
		PS_constantBuffer.data.viewportWidth = width;
		PS_constantBuffer.data.viewportHeight = height;

		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}

	virtual bool release() override
	{
		if (!this->releaseScene())
			return false;
		if (!this->streamlineSolver.release())
			return false;

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