#include "Graphics.h"

#pragma region Main_Initialization
bool Graphics::Initialize(HWND hwnd, int width, int height)
{
	this->windowHeight = height;
	this->windowWidth = width;


	if (!InitializeDirectX(hwnd))
		return false;

	if (!InitializeShaders())
		return false;


	if (!InitializeScene())
		return false;

	//this->InitializeImGui(hwnd);

	//start the timer 
	fpsTimer.Start();


	return true;
}
#pragma endregion Main_Initialization



//########################### Rendering #############################

void Graphics::RenderFrame()
{
	float bgcolor[] = { 0.0f,0.0f, 0.0f, 1.0f };

	
	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view
	this->deviceContext->ClearDepthStencilView(this->depthStencilView.Get(), D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1.0f,0);// Clear the depth stencil view
	this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);	// add depth  stencil state to rendering routin
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());	// Set sampler for the pixel shader
	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);// set the primitive topology
	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);				// set pixel shader

	UINT offset = 0;

	// Set Camera/Eye directions and position
	DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

	// Create the Projection Matrix
	constantBuffer.data.mat = world * camera.GetViewMatrix() * camera.GetProjectionMatrix();
	constantBuffer.data.mat = DirectX::XMMatrixTranspose(constantBuffer.data.mat);

	// Update Constant Buffer
	if (!constantBuffer.ApplyChanges())
	{
		return;
	}
	// set the constant buffer
	this->deviceContext->VSSetConstantBuffers(0, 1, this->constantBuffer.GetAddressOf());
	
	// Draw Square using indices
	this->deviceContext->PSSetShaderResources(0, 1, this->myTexture.GetAddressOf());
	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset);
	this->deviceContext->IASetIndexBuffer(this->indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
	this->deviceContext->DrawIndexed(this->indexBuffer.GetBuffersize(),0,0);

	// calculatin FPS
	static int fpsCounter = 0;
	static std::string fpsString = "FPS : 0";
	fpsCounter += 1;
	if (fpsTimer.GetMilisecondsElapsed() > 1000.0)
	{
		fpsString = "FPS: " + std::to_string(fpsCounter);
		fpsCounter = 0;
		fpsTimer.Restart();
	}

	// Draw text
	this->spriteBatch->Begin();
	spriteFont->DrawString(spriteBatch.get(), StringConverter::StringToWide(fpsString).c_str(), DirectX::XMFLOAT2(0, 0), \
		DirectX::Colors::White, 0.0f, DirectX::XMFLOAT2(1.0, 1.0), DirectX::XMFLOAT2(1.0, 1.0));
	this->spriteBatch->End();

#pragma region IMGUI
	////############# Dear ImGui ####################



	//Create ImGui Test Window
//	ImGui::Begin("Test");
//	ImGui::End();

	////Assemble Together Draw Data
//	ImGui::Render();
	////Render Draw Data
//	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

#pragma endregion IMGUI

	// Present the backbuffer
	this->swapchain->Present(0, NULL);
}



#pragma region DirectX_Initialization
// ################################### Initialize DirectX ################################
bool Graphics::InitializeDirectX(HWND hwnd)
{
	// Get an adapter(normally the first one is the graphics card)
	std::vector<AdapterData> adapters = AdapterReader::GetAdapters();

	if (adapters.size() < 1)
	{
		ErrorLogger::Log("No IDXGI adapter found!");
		return false;
	}

	// Swapchain description structure
	DXGI_SWAP_CHAIN_DESC scd;
	ZeroMemory(&scd,sizeof(DXGI_SWAP_CHAIN_DESC));
	scd.BufferDesc.Width = this->windowWidth;
	scd.BufferDesc.Height = this->windowHeight;
	scd.BufferDesc.RefreshRate.Numerator = 60;
	scd.BufferDesc.RefreshRate.Denominator = 1;
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	scd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	scd.SampleDesc.Count = 1;
	scd.SampleDesc.Quality = 0;
	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scd.BufferCount = 1;
	scd.OutputWindow = hwnd;
	scd.Windowed = TRUE;
	scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	HRESULT hr;

	// Create Device and swapchain
	hr = D3D11CreateDeviceAndSwapChain(
		adapters[0].pAdapter,	//IDXGI adapter (noramlly the first adapter is the hardware and the second one is the software accelarator)
		D3D_DRIVER_TYPE_UNKNOWN,
		NULL,					// For software driver type
		NULL,					// Flags for runtime layers
		NULL,					// Feature levels array
		0,						// Number of feature levels in array
		D3D11_SDK_VERSION,		// SDK version
		&scd,					// Swapchain description
		this->swapchain.GetAddressOf(), // Pointer to the address of swapchain
		this->device.GetAddressOf(),	// Pointer to the address of device
		NULL,					// supported feature level
		this->deviceContext.GetAddressOf() // Pointer to the address of device context
		);

	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Device and Swapchain");
		return false;
	}


	//create and bind the backbuffer
	Microsoft::WRL::ComPtr<ID3D11Texture2D> backbuffer;
	hr = this->swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backbuffer.GetAddressOf()));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Get Back buffer");
		return false;
	}

	// Create Render targe view
	hr = this->device->CreateRenderTargetView(backbuffer.Get(), NULL, this->renderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create RenderTargetView");
		return false;
	}
	//test memory
	backbuffer.Get()->Release();

	// Describe our Depth/Stencil Buffer
	D3D11_TEXTURE2D_DESC depthStencilDesc;
	depthStencilDesc.Width = this->windowWidth;
	depthStencilDesc.Height = this->windowHeight;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilDesc.SampleDesc.Count = 1;
	depthStencilDesc.SampleDesc.Quality = 0;
	depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthStencilDesc.CPUAccessFlags = 0;
	depthStencilDesc.MiscFlags = 0;

	// Create Depth/Stencil buffer
	hr = this->device->CreateTexture2D(&depthStencilDesc, NULL, this->depthStencilBuffer.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Depth/Stencil buffer");
		return false;
	}

	// Create Depth/Stencil View
	hr = this->device->CreateDepthStencilView(this->depthStencilBuffer.Get(), NULL, this->depthStencilView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Depth/Stencil View");
		return false;
	}
	this->deviceContext->OMSetRenderTargets(1, this->renderTargetView.GetAddressOf(), this->depthStencilView.Get());

	// Create depth stencil description structure
	D3D11_DEPTH_STENCIL_DESC depthstencildesc;
	ZeroMemory(&depthstencildesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	depthstencildesc.DepthEnable = true;
	depthstencildesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK::D3D11_DEPTH_WRITE_MASK_ALL;
	depthstencildesc.DepthFunc = D3D11_COMPARISON_FUNC::D3D11_COMPARISON_LESS_EQUAL;

	// Create depth stencil state
	hr = this->device->CreateDepthStencilState(&depthstencildesc, this->depthStencilState.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Depth/Stencil state");
		return false;
	}

	// Create the Viewport
	D3D11_VIEWPORT viewport;
	// Structure to set viewport attributes
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = static_cast<float>(this->windowWidth);
	viewport.Height = static_cast<float>(this->windowHeight);
	viewport.MaxDepth = 1.0f; 
	viewport.MinDepth = 0.0f;

	// Set the Viewport
	this->deviceContext->RSSetViewports(1, &viewport);

	// Create Rasterizer state
	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

	rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_BACK; // CULLING could be set to none
	//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;

	hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
		return false;
	}

	// Initialize the Fonts
	this->spriteBatch	= std::make_unique<DirectX::SpriteBatch>(this->deviceContext.Get());
	this->spriteFont = std::make_unique<DirectX::SpriteFont>(this->device.Get(), L"Data\\Fonts\\comic_sans_ms_16.spritefont");

	// Create Sampler description for sampler state
	D3D11_SAMPLER_DESC sampleDesc;
	ZeroMemory(&sampleDesc, sizeof(D3D11_SAMPLER_DESC));
	sampleDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampleDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampleDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampleDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampleDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	sampleDesc.MinLOD = 0;
	sampleDesc.MaxLOD = D3D11_FLOAT32_MAX;
	
	// Create sampler state
	hr = this->device->CreateSamplerState(&sampleDesc, this->samplerState.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create sampler state.");
		return false;
	}


	return true;
}
#pragma endregion DirectX_Initialization

//########################### Shader Initialization #####################
bool Graphics::InitializeShaders()
{
	std::wstring shaderfolder;
#pragma region DetermineShaderPath
	if (IsDebuggerPresent() == TRUE)
	{
#ifdef _DEBUG //Debug Mode
	#ifdef _WIN64 //x64
		shaderfolder = L"x64\\Debug\\";
	#else //x86
		shaderfolder = L"Debug\\"
	#endif // DEBUG
	#else //Release mode
	#ifdef _WIN64 //x64
				shaderfolder = L"x64\\Release\\";
	#else  //x86
				shaderfolder = L"Release\\"
	#endif // Release
#endif // _DEBUG or Release mode
	}
	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{"POSITION",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,\
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"TEXCOORD",0,DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,\
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0}

	};
	UINT numElements = ARRAYSIZE(layout);

	if (!vertexshader.Initialize(this->device, shaderfolder+L"vertexshader.cso",layout, numElements))
	{
		return false;
	}

	if (!pixelshader.Initialize(this->device, shaderfolder + L"pixelshader.cso"))
	{
		return false;
	}
	
	
	return true;
}

#pragma region Scene_Initialization
// ############################# Initialize the Scene #########################
bool Graphics::InitializeScene()
{
#pragma region Create Vertices
	// Square
	Vertex v[] =
	{
		// left triangle
		Vertex(-0.5f,-0.5f,2.0f,0.0f,1.0f), // Bottom left	-[0]
		Vertex(-0.5f, 0.5f,2.0f,0.0f,0.0f),	// Top left		-[1]
		Vertex( 0.5f, 0.5f,2.0f,1.0f,0.0f),	// Top right	-[2]
		Vertex( 0.5f,-0.5f,2.0f,1.0f,1.0f)	// Bottom right	-[3]
	};
#pragma endregion Create Vertices

#pragma region Create_Indices
	DWORD indices[] = 
	{
		0,1,2,
		0,2,3
	};

#pragma endregion Create_Indices

#pragma region Create_Vertex_Buffer
	// Initialize Vertex Buffer
	HRESULT hr = this->vertexBuffer.Initialize(this->device.Get(), v, ARRAYSIZE(v));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
		return false;
	}
#pragma endregion Create_Vertex_Buffer

#pragma region Create_Index_Buffer
	hr = this->indexBuffer.Initialize(this->device.Get(),indices,ARRAYSIZE(indices));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
		return false;
	}
#pragma endregion Create_Index_Buffer

#pragma region Create_Texture
	// Create Texture View
	hr = DirectX::CreateWICTextureFromFile(this->device.Get(), L"Data\\Textures\\test.jpg", nullptr, this->myTexture.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Texture from file.");
		return false;
	}
#pragma endregion Create_Texture

#pragma region Create_Constant_Buffer
	hr = this->constantBuffer.Initialize(this->device.Get(), this->deviceContext.Get());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Constant buffer.");
		return false;
	}
#pragma endregion Create_Constant_Buffer

	// set camera properties
	camera.SetPosition(0.0f, 0.0f, 0.0f);
	camera.SetProjectionValues(90.0f, static_cast<float>(this->windowWidth) / static_cast<float>(this->windowHeight), \
		0.1f, 1000.0f);

	return true;
}
bool Graphics::InitializeImGui(HWND hwnd)
{
	if (this->IsImGui == false)
	{
		
		IMGUI_CHECKVERSION();
		this->imcontext = ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		// Setup Dear ImGui style
		
		ImGui_ImplWin32_Init(hwnd);
		ImGui_ImplDX11_Init(this->device.Get(), this->deviceContext.Get());

		ImGui::StyleColorsDark();
		OutputDebugStringA("ImGui is created!!!\n");
		this->IsImGui = true;

		return true;
	}

	if (this->IsImGui == true)
	{
		ImGui::SetCurrentContext(this->imcontext);
		OutputDebugStringA("ImGui is created again!!!\n");

		return true;
	}
	
	return true;
}


void Graphics::Resize(HWND hwnd, int width, int height)
{
	this->windowWidth = width;
	this->windowHeight = height;


	this->deviceContext.Get()->OMSetRenderTargets(0, 0, 0);

	// Release all outstanding references to the swap chain's buffers.
	this->renderTargetView.Get()->Release();

	HRESULT hr;
	// Preserve the existing buffer count and format.
	// Automatically choose the width and height to match the client rect for HWNDs.
	hr = this->swapchain.Get()->ResizeBuffers(1, windowWidth, windowHeight, DXGI_FORMAT_UNKNOWN, 0);
	// Perform error handling here!
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Resize the Buffer");
	}
	// Get buffer and create a render-target-view.
	ID3D11Texture2D* pBuffer;
	hr = this->swapchain.Get()->GetBuffer(0, __uuidof(ID3D11Texture2D),
		(void**)&pBuffer);
	// Perform error handling here!
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Filed to Get Buffer");
	}


	hr = this->device.Get()->CreateRenderTargetView(pBuffer, NULL,
		this->renderTargetView.GetAddressOf());
	// Perform error handling here!
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create RenderTargetView");
	}
	pBuffer->Release();

	this->deviceContext.Get()->OMSetRenderTargets(1, this->renderTargetView.GetAddressOf(), NULL);

	// Set up the viewport.
	D3D11_VIEWPORT vp;
	vp.Width = width;
	vp.Height = height;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	this->deviceContext.Get()->RSSetViewports(1, &vp);

}
#pragma endregion Scene_Initialization



