#include "StreamlineRenderer.h"


bool StreamlineRenderer::initializeOITBuffers()
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


	D3D11_BUFFER_DESC BufferDesc;
	BufferDesc.CPUAccessFlags = 0;
	BufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	BufferDesc.ByteWidth = (DWORD)(n_occlusion * width * height *
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


	return true;
}


bool StreamlineRenderer::initializeOITViews()
{
	HRESULT hr;


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

	// Create UAV view of Fragment and Link Buffer
	D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
	UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
	UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	UAVDesc.Buffer.FirstElement = 0;
	UAVDesc.Buffer.NumElements = (DWORD)(n_occlusion * width * height);
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
	SRVBufferDesc.Buffer.ElementWidth = (DWORD)(n_occlusion * width * height);
	hr = device->CreateShaderResourceView(g_pFragmentAndLinkStructuredBuffer.Get(), &SRVBufferDesc,
		&g_pFragmentAndLinkStructuredBufferSRV);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create SRV link list buffer view!");
		return false;
	}




	return true;
}


bool StreamlineRenderer::initializeSampler()
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

	return true;
}

bool StreamlineRenderer::initializeBlendandDepth()
{
	HRESULT hr;


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


	DSDesc.DepthEnable = FALSE;
	DSDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	DSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	DSDesc.StencilEnable = TRUE;
	DSDesc.StencilReadMask = 0xFF;
	DSDesc.StencilWriteMask = 0x00;
	DSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_LESS;
	DSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilFunc = D3D11_COMPARISON_LESS;
	hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestDisabledStencilTestLessDSS);
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
	DSDesc.StencilWriteMask = 0x00;
	DSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_LESS;
	DSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	DSDesc.BackFace.StencilFunc = D3D11_COMPARISON_LESS;
	hr = device->CreateDepthStencilState(&DSDesc, &g_pDepthTestEnabledStencilTestLessDSS);
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




	D3D11_TEXTURE2D_DESC            TexDesc;
	// Create a screen-sized depth stencil resource
	// Use a full 32-bits format for depth when depth peeling is used
	// This is to avoid Z-fighting artefacts due to the "manual" depth buffer implementation
	TexDesc.Width = width;
	TexDesc.Height = height;
	TexDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	TexDesc.SampleDesc.Count = 4;
	TexDesc.SampleDesc.Quality = 0;
	TexDesc.MipLevels = 1;
	TexDesc.Usage = D3D11_USAGE_DEFAULT;
	TexDesc.MiscFlags = 0;
	TexDesc.CPUAccessFlags = 0;
	TexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
	TexDesc.ArraySize = 1;
	hr = device->CreateTexture2D(&TexDesc, NULL, &g_pDepthStencilTexture);

	// Create Depth Stencil View
	D3D11_DEPTH_STENCIL_VIEW_DESC SRVDepthStencilDesc;
	SRVDepthStencilDesc.Format = DXGI_FORMAT_R32_FLOAT;;
	SRVDepthStencilDesc.ViewDimension =D3D11_DSV_DIMENSION_TEXTURE2DMS;
	SRVDepthStencilDesc.Texture2D.MipSlice = 0;
	SRVDepthStencilDesc.Flags = 0;
	hr = device->CreateDepthStencilView(g_pDepthStencilTexture, &SRVDepthStencilDesc,
		&g_pDepthStencilTextureDSV);





	return true;
}




bool StreamlineRenderer::initializeQuadViewPlane()
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

	HRESULT hr = this->vertexBufferQuadViewPlane.Initialize(this->device, quad, ARRAYSIZE(quad));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create offset buffer.");
		return false;
	}


	return true;

}


bool StreamlineRenderer::initializeQuadProjectionPlane()
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

	HRESULT hr = this->vertexBufferQuadProjectionPlane.Initialize(this->device, quad, ARRAYSIZE(quad));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create offset buffer.");
		return false;
	}

	return true;
}



bool StreamlineRenderer::initializeOITRTV()
{
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));

	textureDesc.ArraySize = 1;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.Height = this->height;
	textureDesc.Width = this->width;
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


void StreamlineRenderer::releaseResources()
{

	rasterizerstate.Reset();
	samplerState.Reset();

	g_pStartOffsetBuffer.Reset();
	g_pFragmentAndLinkStructuredBuffer.Reset();

	SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferSRV);
	SAFE_RELEASE(g_pFragmentAndLinkStructuredBufferUAV);
	SAFE_RELEASE(g_pStartOffsetBufferUAV);
	SAFE_RELEASE(g_pStartOffsetBufferSRV);

	SAFE_RELEASE(g_pBlendStateSrcAlphaInvSrcAlphaBlend);
	SAFE_RELEASE(g_pBlendStateNoBlend);
	SAFE_RELEASE(g_pColorWritesOff);

	SAFE_RELEASE(g_pDepthTestEnabledNoDepthWritesStencilWriteIncrementDSS);
	SAFE_RELEASE(g_pDepthTestDisabledDSS);
	SAFE_RELEASE(g_pDepthTestEnabledNoDepthWritesDSS);
	SAFE_RELEASE(g_pDepthTestEnabledDSS);
	SAFE_RELEASE(g_pDepthTestEnabledStencilTestLessDSS);
	SAFE_RELEASE(g_pDepthTestDisabledStencilTestLessDSS);
	SAFE_RELEASE(g_pDepthStencilTextureDSV);
	SAFE_RELEASE(g_pDepthStencilTexture);


	OITTexture.Reset();
	OITRenderTargetView.Reset();
	OITResourceView.Reset();

	vertexBufferQuadProjectionPlane.reset();
	vertexBufferQuadViewPlane.reset();
}



void StreamlineRenderer::updateConstantBuffer(Camera & camera)
{
	//GS_constantBuffer.data.eyePos = DirectX::XMFLOAT3(0, 0, 10);
	//GS_constantBuffer.data.viewDir = DirectX::XMFLOAT3(0, 0 , 1);
	//GS_constantBuffer.data.View = world * camera.GetViewMatrix(DirectX::XMFLOAT3(0, 0, 0));
	//GS_constantBuffer.data.Proj = camera.GetParallelProjectionMatrix();

	DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();
	GS_constantBuffer.data.View = world * camera.GetViewMatrix();
	GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
	GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
	GS_constantBuffer.data.viewDir = camera.GetViewVector();



	GS_constantBuffer.data.tubeRadius = renderingOptions->tubeRadius;
	GS_constantBuffer.data.projection = solverOptions->projection;
	GS_constantBuffer.data.periodicity = solverOptions->periodic;


	GS_constantBuffer.data.gridDiameter.x = fieldOptions->gridDiameter[0];
	GS_constantBuffer.data.gridDiameter.y = fieldOptions->gridDiameter[1];
	GS_constantBuffer.data.gridDiameter.z = fieldOptions->gridDiameter[2];


	if (solverOptions->projection == Projection::STREAK_PROJECTION)
	{
		GS_constantBuffer.data.particlePlanePos = streakProjectionPlane();
	}
	else if (solverOptions->projection == Projection::STREAK_PROJECTION_FIX)
	{
		GS_constantBuffer.data.particlePlanePos = -solverOptions->timeDim / 2;;
		GS_constantBuffer.data.particlePlanePos = 0;
	}

	GS_constantBuffer.data.streakPos = solverOptions->projectPos * (fieldOptions->gridDiameter[0] / fieldOptions->gridSize[0]);
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
	PS_constantBuffer.data.Ka = renderingOptions->Ka;
	PS_constantBuffer.data.Kd = renderingOptions->Kd;
	PS_constantBuffer.data.Ks = renderingOptions->Ks;
	PS_constantBuffer.data.shininessVal = renderingOptions->shininess;


		// Update Constant Buffer
	GS_constantBuffer.ApplyChanges();
	PS_constantBuffer.ApplyChanges();
	
}

void StreamlineRenderer::updateBuffersSecondPass()
{
	PS_constantBufferSampler.data.viewportHeight = height;
	PS_constantBufferSampler.data.viewportWidth = width;
	PS_constantBufferSampler.ApplyChanges();

}