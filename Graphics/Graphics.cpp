#include "Graphics.h"
#include "RenderImGui.h"



#pragma region Main_Initialization
bool Graphics::Initialize(HWND hwnd, int width, int height)
{
	this->windowHeight = height;
	this->windowWidth = width;

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

bool Graphics::InitializeRayCastingTexture()
{
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));

	textureDesc.ArraySize = 1;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.Height = windowHeight;
	textureDesc.Width = windowWidth;
	textureDesc.MipLevels = 4;
	textureDesc.MiscFlags = 0;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.SampleDesc.Quality = 0;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;

	HRESULT hr = this->device->CreateTexture2D(&textureDesc, nullptr, this->frontTex.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Front Texture");
	}

	// Create Render targe view

	CD3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
	ZeroMemory(&renderTargetViewDesc, sizeof(CD3D11_RENDER_TARGET_VIEW_DESC));

	renderTargetViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

	hr = this->device->CreateRenderTargetView(frontTex.Get(), NULL, this->viewtofrontText.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create viewtofrontText");
	}

	return true;
	
}

//########################### Rendering #############################

void Graphics::RenderFrame()
{
	float bgcolor[] = { 0.0f,0.0f, 0.0f, 0.0f };
	float bgcolor2[] = { 0.0f,0.0f, 0.0f, 0.0f };

	

	if (this->solverOptions.userInterruption)
	{
		this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), bgcolor);// Clear the target view
	}


	raycastingRendering();
	


	//####################################################### line rendering #####################################################


	this->deviceContext->ClearDepthStencilView(this->depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);// Clear the depth stencil view
	this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);	// add depth  stencil state to rendering routin
	//this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());	// Set sampler for the pixel shader
	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP);// set the primitive topology
	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	//this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);			// set Geometry shader
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



	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); // set Vertex buffer
	this->deviceContext->IASetIndexBuffer(this->indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0); // Set index buffer





	if (showLines)
	{
		//for (int i = 0; i < solverOptions.lines_count; i++)
	//	{
			this->deviceContext->DrawIndexed(llInt(solverOptions.lineLength) * llInt(solverOptions.lines_count) + solverOptions.lines_count, 0, 0);

		//}
	}
	else
	{
		this->deviceContext->Draw(0, 0);

	}
	   


	//######################################### Dear ImGui ######################################

	RenderImGui renderImGui;
	renderImGui.drawSolverOptions(this->solverOptions);
	renderImGui.drawLog(this);
	renderImGui.render();

	
	if (this->solverOptions.beginStream || this->solverOptions.beginPath)
	{
		this->InitializeScene();
	}



	// Present the backbuffer
	this->swapchain->Present(0, NULL);

	// Always turn the user interruption to false after the presentation of backbuffer
	this->solverOptions.userInterruption = false;

	
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
	depthStencilDesc.MipLevels = 0;
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

	D3D_FEATURE_LEVEL feature;

	// Create Device and swapchain
	hr = D3D11CreateDeviceAndSwapChain(
		adapters[0].pAdapter,	//IDXGI adapter (noramlly the first adapter is the hardware and the second one is the software accelarator)
		D3D_DRIVER_TYPE_UNKNOWN,
		NULL,						// For software driver type
		D3D11_CREATE_DEVICE_DEBUG,	// Flags for runtime layers
		NULL,						// Feature levels array
		0,						// Number of feature levels in array
		D3D11_SDK_VERSION,		// SDK version
		&scd,					// Swapchain description
		this->swapchain.GetAddressOf(), // Pointer to the address of swapchain
		this->device.GetAddressOf(),	// Pointer to the address of device
		&feature,					// supported feature level
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
		{"TANGENT",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"LINEID",0,DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"COLOR",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0}
	};
	UINT numElements = ARRAYSIZE(layout);

	if (!vertexshader.Initialize(this->device, shaderfolder+L"vertexshader.cso",layout, numElements))
	{
		return false;
	}

	if (!geometryshader.Initialize(this->device, shaderfolder + L"geometryshader.cso"))
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

	std::vector<DWORD> indices(llInt(solverOptions.lineLength)* llInt(solverOptions.lines_count) + solverOptions.lines_count);
	for (int i = 0; i < indices.size(); i++)
	{
		if (i % (solverOptions.lines_count) == 0)
		{
			indices[i] = -1;
		}
		else
		{
			indices[i] = i;
		}
		
	}

#pragma endregion Create_Indices

#pragma region Create_Vertex_Buffer


	HRESULT hr;

	// Initialize Vertex Buffer

	hr = this->vertexBuffer.Initialize(this->device.Get(), NULL, sizeof(Vertex)*solverOptions.lineLength * solverOptions.lines_count);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
		return false;
	}

#pragma endregion Create_Vertex_Buffer


#pragma region Create_Index_Buffer

		hr = this->indexBuffer.Initialize(this->device.Get(), &indices.at(0), indices.size());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
			return false;
		}


#pragma endregion Create_Index_Buffer




#pragma region Create_Constant_Buffer
	hr = this->constantBuffer.Initialize(this->device.Get(), this->deviceContext.Get());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Constant buffer.");
		return false;
	}
#pragma endregion Create_Constant_Buffer

	// set camera properties
	camera.SetPosition(0.0f, 0.0f, -10.0f);
	camera.SetProjectionValues(this->FOV, static_cast<float>(this->windowWidth) / static_cast<float>(this->windowHeight), \
		1.0f, 100.0f);

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
	this->depthStencilView->Release();
	this->renderTargetView->Release();


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

	// Create Render targe view
	//hr = this->device->CreateRenderTargetView(frontTex.Get(), NULL, this->viewtofrontText.GetAddressOf());
	//if (FAILED(hr))
	//{
	//	ErrorLogger::Log(hr, "Failed to Create viewtofrontText");
	//}


}
#pragma endregion Scene_Initialization


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




const float3 Graphics::getUpVector()
{
	XMFLOAT3 upDir = this->camera.GetUpVector();
	float3 up = { upDir.x,upDir.y,upDir.z };
	
	return up;
}
const float3 Graphics::getEyePosition()
{
	XMFLOAT3 eyePosition = this->camera.GetPositionFloat3();
	float3 eye = { eyePosition.x,eyePosition.y,eyePosition.z };

	return eye;
}
const float3 Graphics::getViewDir()
{

	XMFLOAT3 viewDir =  this->camera.GetViewVector();
	float3 view = { viewDir.x,viewDir.y,viewDir.z };

	return view;
}

bool Graphics::InitializeRaytracingInteroperability()
{
	// define interoperation descriptor and set it to zero
	Interoperability_desc interoperability_desc;
	memset(&interoperability_desc, 0, sizeof(interoperability_desc));

	// set interoperation descriptor
	interoperability_desc.flag = cudaGraphicsRegisterFlagsSurfaceLoadStore;
	interoperability_desc.p_adapter = this->adapter;
	interoperability_desc.p_device = this->device.Get();
	interoperability_desc.size = sizeof(float) *static_cast<size_t>(this->windowWidth) * static_cast<size_t>(this->windowHeight);
	interoperability_desc.pD3DResource = this->getBackbuffer();

	// initialize the interoperation
	cudaRayTracingInteroperability.setInteroperability_desc(interoperability_desc);
	
	return cudaRayTracingInteroperability.InitializeResource();

}


bool Graphics::initializeRaycasting()
{
	// Create and initializes the dst texture for raycasting
	//this->InitializeRayCastingTexture();
	
	// A pointer to the cuda array
	cudaArray_t pCudaArray = NULL;

	// Initialize the interoperation 
	//		+ Set interoperability description such as device pointer and size and the texture to share with CUDA
	//		+ Register and map the D3D Resource to CUDA
	if (!this->InitializeRaytracingInteroperability())
		return false;

	// Get the an array to the D3D resource and cast it to a cudaArray
	cudaRayTracingInteroperability.getMappedArray(pCudaArray);

	// Pass the CudaArray to the Surface Array
	cudaSurface.setInputArray(pCudaArray);

	// Now initialize the Surface
	if (!cudaSurface.initializeSurface())
		return false;

	return true;
}


void Graphics::raycastingRendering()
{
	if (this->solverOptions.beginRaycasting)
	{
		if (this->solverOptions.idChange)
		{
			/*std::string upDirection = "";
		upDirection += std::to_string(getViewDir().x);
		upDirection += ", ";
		upDirection += std::to_string(getViewDir().y);
		upDirection += ", ";
		upDirection += std::to_string(getViewDir().z);
		upDirection += "\n";
		OutputDebugStringA(upDirection.c_str());
*/
			Raycasting_desc raycasting_desc;
			ZeroMemory(&raycasting_desc, sizeof(raycasting_desc));

			raycasting_desc.width = this->getWindowWidth();
			raycasting_desc.height = this->getWindowHeight();
			raycasting_desc.gridDiameter = make_float3(this->solverOptions.gridDiameter[0], this->solverOptions.gridDiameter[1], this->solverOptions.gridDiameter[2]);
			raycasting_desc.gridSize = make_int3(this->solverOptions.gridSize[0], this->solverOptions.gridSize[1], this->solverOptions.gridSize[2]);
			raycasting_desc.viewDir = this->getViewDir();
			raycasting_desc.eyePos = this->getEyePosition();
			raycasting_desc.FOV_deg = this->getFOV();
			raycasting_desc.upDir = this->getUpVector();
			raycasting_desc.solverOption = this->solverOptions;

			raycasting.setRaycastingDec(raycasting_desc);

			// Intializes Interopration: bind texture to cuda array, create a surface and bind the cuda array to it
			this->initializeRaycasting();

			// Now bind the cuda surface to the raycaster
			raycasting.setRaycastingSurface(this->getSurfaceObject());


			// + Create a bounding box and initialize it
			// + Copy bounding box to GPU
			// + Create and copy rays to GPU
			raycasting.initialize();

			// + Run the Cuda Kernel
			raycasting.Rendering();

			// + Release bounding box and rays
			raycasting.release();

			//std::string fileName = "test.dds";
			//std::wstring wfileName = std::wstring(fileName.begin(), fileName.end());
			//const wchar_t* result = wfileName.c_str();
			//SaveDDSTextureToFile(this->gfx.GetDeviceContext(), this->gfx.getTexture(), result);

			this->cudaSurface.destroySurface();

			// Unmap and Unregister the D3D resource
			this->cudaRayTracingInteroperability.release();

			this->solverOptions.idChange = false;
		}
		else if (this->solverOptions.userInterruption)
		{
			Raycasting_desc raycasting_desc;
			ZeroMemory(&raycasting_desc, sizeof(raycasting_desc));

			raycasting_desc.width = this->getWindowWidth();
			raycasting_desc.height = this->getWindowHeight();
			raycasting_desc.gridDiameter = make_float3(this->solverOptions.gridDiameter[0], this->solverOptions.gridDiameter[1], this->solverOptions.gridDiameter[2]);
			raycasting_desc.gridSize = make_int3(this->solverOptions.gridSize[0], this->solverOptions.gridSize[1], this->solverOptions.gridSize[2]);
			raycasting_desc.viewDir = this->getViewDir();
			raycasting_desc.eyePos = this->getEyePosition();
			raycasting_desc.FOV_deg = this->getFOV();
			raycasting_desc.upDir = this->getUpVector();
			raycasting_desc.solverOption = this->solverOptions;

			raycasting.setRaycastingDec(raycasting_desc);

			// Intializes Interopration: bind texture to cuda array, create a surface and bind the cuda array to it
			this->initializeRaycasting();

			// Now bind the cuda surface to the raycaster
			raycasting.setRaycastingSurface(this->getSurfaceObject());


			// + Create a bounding box and initialize it
			// + Copy bounding box to GPU
			// + Create and copy rays to GPU
			raycasting.initialize();

			// + Run the Cuda Kernel
			raycasting.Rendering();

			// + Release bounding box and rays
			raycasting.release();

			//std::string fileName = "test.dds";
			//std::wstring wfileName = std::wstring(fileName.begin(), fileName.end());
			//const wchar_t* result = wfileName.c_str();
			//SaveDDSTextureToFile(this->gfx.GetDeviceContext(), this->gfx.getTexture(), result);

			this->cudaSurface.destroySurface();

			// Unmap and Unregister the D3D resource
			this->cudaRayTracingInteroperability.release();


		}


	}
}