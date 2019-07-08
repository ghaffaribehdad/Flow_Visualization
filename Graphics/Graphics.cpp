#include "Graphics.h"




#pragma region Main_Initialization
bool Graphics::Initialize(HWND hwnd, int width, int height)
{
	this->windowHeight = height;
	this->windowWidth = width;
	strcpy(log, "Initialized");

	if (!this->InitializeDirectX(hwnd))
		return false;

	if (!this->InitializeResources())
		return false;

	if (!this->InitializeShaders())
		return false;


	if (!this->InitializeScene())
		return false;

	if (!this->InitializeImGui(hwnd))
		return false;

	//start the timer 
	fpsTimer.Start();

	return true;
}
#pragma endregion Main_Initialization



//########################### Rendering #############################

void Graphics::RenderFrame()
{
	float bgcolor[] = { 0.0f,0.0f, 0.0f, 0.0f };


	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view
	this->deviceContext->ClearDepthStencilView(this->depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);// Clear the depth stencil view
	this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);	// add depth  stencil state to rendering routin
	//this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());	// Set sampler for the pixel shader
	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP);// set the primitive topology
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
	//set the constant buffer
	this->deviceContext->VSSetConstantBuffers(0, 1, this->constantBuffer.GetAddressOf());

	// Draw lines using indices
	//this->deviceContext->PSSetShaderResources(0, 1, this->myTexture.GetAddressOf());



	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset);


	this->deviceContext->IASetIndexBuffer(this->indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
	if (showLines)
	{
		for (int i = 0; i < solverOptions.lines_count; i++)
		{
			this->deviceContext->DrawIndexed(solverOptions.lineLength, i* solverOptions.lineLength, 0);

		}
	}
	else
	{
		this->deviceContext->Draw(0, 0);

	}
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

	//######################################### Dear ImGui ######################################
	RenderImGui();

	// Present the backbuffer
	this->swapchain->Present(0, NULL);
}


bool Graphics::InitializeResources()
{

	HRESULT hr;

	//create and bind the backbuffer
	Microsoft::WRL::ComPtr<ID3D11Texture2D> backbuffer;
	hr = this->swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backbuffer.GetAddressOf()));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Get Back buffer");
	}

	// Create Render targe view
	hr = this->device->CreateRenderTargetView(backbuffer.Get(), NULL, this->renderTargetView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create RenderTargetView");
	}


	this->deviceContext->OMSetRenderTargets(1, renderTargetView.GetAddressOf(), NULL);


	if (this->depthStencilBuffer != NULL)
	{
		depthStencilBuffer->Release();
	}

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

	hr = this->device->CreateDepthStencilView(this->depthStencilBuffer.Get(), NULL, this->depthStencilView.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Depth/Stencil View");
		return false;
	}
	this->deviceContext->OMSetRenderTargets(1, this->renderTargetView.GetAddressOf(), this->depthStencilView.Get());

	if (this->depthStencilState.Get() == NULL)
	{
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

	if (this->rasterizerstate.Get() == nullptr)
	{
		// Create Rasterizer state
		D3D11_RASTERIZER_DESC rasterizerDesc;
		ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_BACK; // CULLING could be set to none
		rasterizerDesc.MultisampleEnable = false;
		rasterizerDesc.AntialiasedLineEnable = true;
		//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;

		hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
			return false;
		}
	}

	if (this->spriteBatch.get() == nullptr)
	{
		// Initialize the Fonts
		this->spriteBatch = std::make_unique<DirectX::SpriteBatch>(this->deviceContext.Get());
		this->spriteFont = std::make_unique<DirectX::SpriteFont>(this->device.Get(), L"Data\\Fonts\\comic_sans_ms_16.spritefont");
	}

	///if (this->samplerState.Get() == nullptr)
	///{
	///	// Create Sampler description for sampler state
	///	D3D11_SAMPLER_DESC sampleDesc;
	///	ZeroMemory(&sampleDesc, sizeof(D3D11_SAMPLER_DESC));
	///	sampleDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	///	sampleDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	///	sampleDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	///	sampleDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	///	sampleDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	///	sampleDesc.MinLOD = 0;
	///	sampleDesc.MaxLOD = D3D11_FLOAT32_MAX;

	///	// Create sampler state
	///	hr = this->device->CreateSamplerState(&sampleDesc, this->samplerState.GetAddressOf());
	///	if (FAILED(hr))
	///	{
	///		ErrorLogger::Log(hr, "Failed to Create sampler state.");
	///		return false;
	///}
	///}

	return true;
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

	this->adapter = adapters[0].pAdapter;

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
		D3D11_CREATE_DEVICE_DEBUG,	// Flags for runtime layers
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
		{"VELOCITY",0,DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,\
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"TANGENT",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,\
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

	//Vertex v[] =
	//{
	//	// left triangle
	//	Vertex(-0.5f,-0.5f,1.0f,0.1f,1.0f), // Bottom left	-[0]
	//	Vertex(-0.5f, 0.5f,0.0f,0.50f,0.0f),	// Top left		-[1]
	//	Vertex(0.7f, 0.5f,1.0f,1.0f,0.0f),	// Top right	-[2]
	//	Vertex(0.5f,-0.5f,1.0f,1.0f,1.0f)	// Bottom right	-[3]
	//};



#pragma endregion Create Vertices


#pragma region Create_Indices

	std::vector<DWORD> indices(solverOptions.lineLength*solverOptions.lines_count);
	for (int i = 0; i < indices.size(); i++)
	{
		indices[i] = i;
	}

#pragma endregion Create_Indices

#pragma region Create_Vertex_Buffer


	HRESULT hr;

	// Initialize Vertex Buffer
	if (this->vertexBuffer.Get() == NULL)
	{
		hr = this->vertexBuffer.Initialize(this->device.Get(), NULL, sizeof(Vertex)*solverOptions.lineLength * solverOptions.lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}
	}
	else
	{
		hr = this->vertexBuffer.Initialize(this->device.Get(), NULL, sizeof(Vertex) * solverOptions.lineLength * solverOptions.lines_count);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}
	}

#pragma endregion Create_Vertex_Buffer


#pragma region Create_Index_Buffer
	if (this->indexBuffer.Get() == NULL)
		{
			hr = this->indexBuffer.Initialize(this->device.Get(), &indices.at(0), indices.size());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
			return false;
		}
	}
	else
	{

		hr = this->indexBuffer.Initialize(this->device.Get(), &indices.at(0), indices.size());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
			return false;
		}
	}

#pragma endregion Create_Index_Buffer

#pragma region Create_Texture
	/// Create Texture View
	///if (this->myTexture.Get() == nullptr)
	///{
	///	hr = DirectX::CreateWICTextureFromFile(this->device.Get(), L"Data\\Textures\\test.jpg", nullptr, this->myTexture.GetAddressOf());
	///	if (FAILED(hr))
	///	{
	///		ErrorLogger::Log(hr, "Failed to Create Texture from file.");
	///		return false;
	///	}
	///}

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
		0.1f, 100.0f);

	return true;
}
bool Graphics::InitializeImGui(HWND hwnd)
{
	if (this->ImGuicontext == nullptr)
	{

		OutputDebugStringA("ImGui is created!!!\n");
		IMGUI_CHECKVERSION();
		this->ImGuicontext = ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		// Setup Dear ImGui style
		ImGui_ImplWin32_Init(hwnd);
		ImGui_ImplDX11_Init(this->device.Get(), this->deviceContext.Get());
		ImGui::StyleColorsDark();

		return true;
	}
	else
	{
		return true;
	}
	
}


void Graphics::Resize(HWND hwnd)
{
	// Retrieve the coordinates of a window's client area. 
	RECT rect;
	GetClientRect(hwnd, &rect);
	this->windowWidth = rect.right - rect.left;
	this->windowHeight = rect.bottom - rect.top;

	// set Render target to NULL
	this->deviceContext->OMSetRenderTargets(0, 0, 0);

	// Release RenderTarge View nad Depth Stencil View
	this->renderTargetView->Release();
	this->depthStencilView->Release();

	// Resize the swapchain
	HRESULT hr = this->swapchain->ResizeBuffers(1, windowWidth, windowHeight, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed Resize Swapchain buffers");
	}

	// Create and bind the backbuffer
	Microsoft::WRL::ComPtr<ID3D11Texture2D> backbuffer;
	hr = this->swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backbuffer.GetAddressOf()));
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Get Back buffer");
	}
	
	// Reinitialize Resources and Scene accordingly
	this->InitializeResources();
	this->InitializeScene(); 

}
#pragma endregion Scene_Initialization



void Graphics::RenderImGui()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

#pragma region Solver_Option

	ImGui::Begin("Solver Options");

	ImGui::Text("Mode: ");
	ImGui::SameLine();

	//Solver Mode

	if (ImGui::Checkbox("Streamline", &streamline))
	{
		pathline = !streamline;
		this->solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		this->InitializeScene();
	}
	ImGui::SameLine();
	if (ImGui::Checkbox("Pathline", &pathline))
	{
		streamline = !pathline;
		this->solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		this->InitializeScene();
	}


	// Solver Options
	if (ImGui::InputText("File Path", this->solverOptions.filePath, sizeof(this->solverOptions.filePath)))
	{
	}

	if (ImGui::InputText("File Name", this->solverOptions.fileName, sizeof(this->solverOptions.fileName)))
	{
	}

	if (ImGui::InputInt3("Grid Size", this->solverOptions.gridSize, sizeof(this->solverOptions.gridSize)))
	{
	}
	if (ImGui::InputFloat3("Grid Diameter", this->solverOptions.gridDiameter, sizeof(this->solverOptions.gridDiameter)))
	{
	}

	if (ImGui::InputInt("precision", &(this->solverOptions.precision)))
	{
	
	}
	ImGui::PushItemWidth(75);
	if (ImGui::InputInt("First Index", &(this->solverOptions.firstIdx)))
	{

	}
	ImGui::SameLine();
	if (ImGui::InputInt("Last Index", &(this->solverOptions.lastIdx)))
	{

	}
	if (ImGui::DragInt("Current Index", &(this->solverOptions.currentIdx), 1.0f, 0, solverOptions.lastIdx, "%d"))
	{

	}
	ImGui::PopItemWidth();
	if (ImGui::InputFloat("dt", &(this->solverOptions.dt)))
	{

	}

	if (ImGui::InputInt("Lines", &(this->solverOptions.lines_count)))
	{
		if(this->pathline)
		{
			this->solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		}
		this->InitializeScene();


	}
	if (this->streamline)
	{
		if (ImGui::InputInt("Line Length", &(this->solverOptions.lineLength)))
		{
			this->InitializeScene();

		}
	}
	if (ImGui::ListBox("Color Mode",&this->solverOptions.colorMode,ColorModeList,4))
	{

	}

	if (this->streamline)
	{
		if (ImGui::Checkbox("Render Streamlines", &this->solverOptions.beginStream))
		{
			strcpy(log, "It is running!");
		}

	}
	else
	{
		if (ImGui::Checkbox("Render Pathlines", &this->solverOptions.beginPath))
		{
			strcpy(log, "It is running!");
		}
	}



	ImGui::End();
#pragma endregion Solver_Options

#pragma region Log

	ImGui::Begin("Log");
	

	if (ImGui::InputTextMultiline("Log",this->log,1000))
	{

	}





	ImGui::End();
#pragma endregion Log


	//Assemble Together Draw Data
	ImGui::Render();

	//Render Draw Data
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

IDXGIAdapter* Graphics::GetAdapter()
{
	return adapter;
}

ID3D11Device* Graphics::GetDevice()
{
	return this->device.Get();
}

ID3D11Buffer* Graphics::GetVertexBuffer()
{
	return this->vertexBuffer.Get();
}