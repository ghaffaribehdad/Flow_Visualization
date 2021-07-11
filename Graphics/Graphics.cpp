#include "Graphics.h"
#include <ScreenGrab11.h>
#include <dxgidebug.h>
#include <dxgi1_3.h>
#include <wincodec.h>
#include "GraphicsHelper.h"


bool Graphics::InitializeCamera()
{
	// set camera properties
	camera.SetPosition(cameraProp.eyePosition);
	camera.SetProjectionValues(
		this->cameraProp.FOV,
		static_cast<float>(this->windowWidth) / static_cast<float>(this->windowHeight),
		this->cameraProp.nearField,
		this->cameraProp.farField
	);
	camera.SetParallelProjectionValues(
		static_cast<float>(this->windowWidth) / static_cast<float>(this->windowHeight),
		static_cast<float>(this->windowHeight),
		this->cameraProp.nearField,
		this->cameraProp.farField
	);
	return true;
}



#pragma region Main_Initialization
bool Graphics::Initialize(HWND hwnd, int width, int height)
{
	this->windowHeight = height;
	this->windowWidth = width;

	if (!this->InitializeDirectX(hwnd))
		return false;


	if (!this->InitializeDirectXResources())
		return false;

	if (!this->InitializeResources())
		return false;

	if (!this->InitializeShaders())
		return false;

	if (!this->InitializeCamera())
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



void Graphics::RenderFrame()
{

	if (renderImGuiOptions.updateShaders)
	{
		InitializeShaders();
		renderImGuiOptions.updateShaders = false;

	}

	if (renderImGuiOptions.saveScreenshot)
	{
		// Create RT and RTV
		createTexture(2560, 1377, device.Get(), texture_screenshot.GetAddressOf());
		this->device->CreateRenderTargetView(texture_screenshot.Get(), NULL, rtv_Screenshot.GetAddressOf());
		
		// Create Depth Stencil and Depth Stencil View
		createTexture(2560, 1377, device.Get(), depthStencilBuffer_Screenshot.GetAddressOf(), D3D11_BIND_DEPTH_STENCIL, { 4,0 }, DXGI_FORMAT_D24_UNORM_S8_UINT);
		this->device->CreateDepthStencilView(this->depthStencilBuffer_Screenshot.Get(), NULL, this->depthStencilView_Screenshot.GetAddressOf());
		this->deviceContext->ClearRenderTargetView(this->rtv_Screenshot.Get(), renderingOptions.bgColor);
		this->deviceContext->ClearDepthStencilView(this->depthStencilView_Screenshot.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
		this->deviceContext->OMSetRenderTargets(1, this->rtv_Screenshot.GetAddressOf(), this->depthStencilView_Screenshot.Get());
		this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);

	}
	else
	{
		this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions.bgColor);
		this->deviceContext->ClearDepthStencilView(this->depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
		this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);
		//this->deviceContext->OMSetRenderTargets(1, this->renderTargetView.GetAddressOf(), this->depthStencilView_Screenshot.Get());

	}


	this->mouseSpeed = this->renderingOptions.mouseSpeed;

	/*

##############################################################
##															##
##						Show Scene							##
##															##
##############################################################

*/


	volumeBox.show(&renderImGuiOptions, solverOptions.gridDiameter);
	seedBox.show(&renderImGuiOptions, solverOptions.seedBox, solverOptions.seedBoxPos);
	clipBox.show(&renderImGuiOptions, raycastingOptions.clipBox, raycastingOptions.clipBoxCenter);
	streakBox.show(&renderImGuiOptions, solverOptions.streakBox, solverOptions.streakBoxPos);
	streakPlane.show(&renderImGuiOptions, solverOptions.streakBox, solverOptions.streakBoxPos);



	if (!renderImGuiOptions.pauseRendering)
	{
		streamlineRenderer.show(&renderImGuiOptions);		// Streamline rendering
		streaklineRenderer.show(&renderImGuiOptions);		// Streakline rendering
		pathlineRenderer.show(&renderImGuiOptions);			// Pathline rendering
		fluctuationHeightfield.show(&renderImGuiOptions);	// Fluctuation Heightfield
	}


	/*
	##############################################################
	##															##
	##							Draw							##
	##															##
	##############################################################
	*/
	if (renderingOptions.showVolumeBox)
	{
		this->volumeBox.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
	}

	if (renderingOptions.showSeedBox)
	{
		this->seedBox.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	}


	if (renderingOptions.showStreakBox)
	{
		this->streakBox.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	}



	if (renderingOptions.showClipBox)
	{
		this->clipBox.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

	}


	if (renderingOptions.showStreakPlane)
	{
		this->streakPlane.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	}


	raycasting.show(&renderImGuiOptions);					// Raycasting 


	if (this->renderImGuiOptions.showStreaklines)
	{

		this->streaklineRenderer.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ);

	}

	if (this->renderImGuiOptions.showPathlines)
	{
		switch (renderingOptions.renderingMode)
		{
		case RenderingMode::RenderingMode::TUBES:
		{
			this->pathlineRenderer.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ);
			break;
		}
		case RenderingMode::RenderingMode::SPHERES:
		{
			this->pathlineRenderer.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
			break;
		}
		}
	}

	if (renderImGuiOptions.saveScreenshot)
	{
		streamlineRenderer.mainRTV = rtv_Screenshot.Get();
		streamlineRenderer.depthstencil = depthStencilView_Screenshot.Get();
	}
	else
	{
		streamlineRenderer.mainRTV = renderTargetView.Get();
		streamlineRenderer.depthstencil = depthStencilView.Get();
	}


	if (this->renderImGuiOptions.showStreamlines)
	{
		switch (renderingOptions.renderingMode)
		{
		case RenderingMode::RenderingMode::TUBES:
		{
			this->streamlineRenderer.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ);
			break;
		}
		case RenderingMode::RenderingMode::SPHERES:
		{
			this->streamlineRenderer.draw(camera, D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
			break;
		}
		}
	}


	if (renderImGuiOptions.saveScreenshot)
	{
		this->deviceContext->OMSetRenderTargets(1, this->rtv_Screenshot.GetAddressOf(), this->depthStencilView_Screenshot.Get());
	}
	else
	{
		this->deviceContext->OMSetRenderTargets(1, this->renderTargetView.GetAddressOf(), this->depthStencilView.Get());
	}

	this->deviceContext->OMSetDepthStencilState(this->depthStencilState.Get(), 0);




	/*

##############################################################
##															##
##							Release 						##
##															##
##############################################################

*/

	if (!renderImGuiOptions.pauseRendering)
	{
		if (this->renderImGuiOptions.releaseStreamlines)
		{
			streamlineRenderer.release();
			this->renderImGuiOptions.releaseStreamlines = false;
			this->solverOptions.fileLoaded = false;
		}

		if (this->renderImGuiOptions.releasePathlines)
		{
			pathlineRenderer.release();
			this->renderImGuiOptions.releasePathlines = false;

		}

		if (this->renderImGuiOptions.releaseStreaklines)
		{
			streaklineRenderer.release();
			this->renderImGuiOptions.releaseStreaklines = false;
		}
	}


	if (renderImGuiOptions.saveScreenshot)
	{
		std::string fullName = dispersionOptions.filePath + solverOptions.outputFileName + std::to_string(solverOptions.currentIdx) + std::string(".jpg");
		//this->saveTextureJPEG(getBackBuffer(), fullName);
		this->saveTextureJPEG(texture_screenshot.Get(), fullName);
	}


	/*
##############################################################
##															##
##						Dear ImGui							##
##															##
##############################################################
*/

	if (renderImGuiOptions.saveScreenshot)
	{
		renderImGuiOptions.screenshotCounter++;

		if (renderImGuiOptions.screenshotRange == renderImGuiOptions.screenshotCounter)
		{
			renderImGuiOptions.saveScreenshot = false;
			renderImGuiOptions.screenshotCounter = 0;
			releaseScreenshotResources();
			resetRTV();
		}
		else if (solverOptions.lineRenderingMode == LineRenderingMode::STREAMLINES)
		{
			solverOptions.currentIdx++;
			renderImGuiOptions.updateRaycasting = true;
			renderImGuiOptions.updateStreamlines = true;
			raycastingOptions.fileChanged = true;
			solverOptions.fileChanged = true;
			solverOptions.loadNewfile = true;
		}
	}


	renderImGuiOptions.drawOptionWindows();		// Draw Options
	if (!renderImGuiOptions.hideOptions)
	{
		renderImGuiOptions.render();				// Render ImGui 
	}


	// Present the backbuffer
	this->swapchain->Present(1, NULL);

	


}

void Graphics::resetRTV()
{



}

void Graphics::releaseScreenshotResources()
{
	this->depthStencilView_Screenshot.Reset();
	this->depthStencilBuffer_Screenshot.Reset();
	this->depthStencilState_Screenshot.Reset();
	this->texture_screenshot.Reset();
	this->rtv_Screenshot.Reset();
	

}

bool Graphics::InitializeDirectXResources()
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


	depthStencilBuffer.Reset();


	// Describe our Depth/Stencil Buffer
	D3D11_TEXTURE2D_DESC depthStencilDesc;
	depthStencilDesc.Width = this->windowWidth;
	depthStencilDesc.Height = this->windowHeight;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilDesc.SampleDesc.Count = 4;
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

	depthStencilState.Reset();

	// Create depth stencil description structure
	D3D11_DEPTH_STENCIL_DESC depthstencildesc;
	ZeroMemory(&depthstencildesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	depthstencildesc.DepthEnable = true;
	depthstencildesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK::D3D11_DEPTH_WRITE_MASK_ALL;
	depthstencildesc.DepthFunc = D3D11_COMPARISON_FUNC::D3D11_COMPARISON_LESS;

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
	viewport.MaxDepth = 1.0;
	viewport.MinDepth = 0.0f;

	// Set the Viewport
	this->deviceContext->RSSetViewports(1, &viewport);



	return true;
}

bool Graphics::InitializeResources()
{
#if defined(_DEBUG)
	void** m_d3dDebug;
	device->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(&m_d3dDebug));

	Microsoft::WRL::ComPtr<IDXGIDebug1> dxgiDebug;
	DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug));
	dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
#endif
	streamlineRenderer.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter,
		this->windowWidth,
		this->windowHeight
	);

	streaklineRenderer.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);


	pathlineRenderer.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	volumeBox.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	seedBox.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	clipBox.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	streakBox.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	streakPlane.setResources
	(
		this->renderingOptions,
		this->solverOptions,
		this->deviceContext.Get(),
		this->device.Get(),
		this->adapter
	);

	raycasting.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this ->deviceContext.Get()
	);

	crossSection.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->crossSectionOptions
	);

	turbulentMixing.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->turbulentMixingOptions
	);


	fluctuationHeightfield.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->dispersionOptions,
		&this->spaceTimeOptions
	);

	dispersionTracer.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->dispersionOptions
	);

	heightfieldFTLE.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->dispersionOptions
	);


	timeSpacefield.setResources
	(
		&this->camera,
		&this->windowWidth,
		&this->windowHeight,
		&this->solverOptions,
		&this->raycastingOptions,
		&this->renderingOptions,
		this->device.Get(),
		this->adapter,
		this->deviceContext.Get(),
		&this->timeSpace3DOptions
	);

	heightfieldFTLE.fsleOptions = &fsleOptions;
	
	if (!streamlineRenderer.initializeBuffers())
		return false;

	if (!streaklineRenderer.initializeBuffers())
		return false;
	
	if (!pathlineRenderer.initializeBuffers())
		return false;

	if (!volumeBox.initializeBuffers())
		return false;	

	if (!seedBox.initializeBuffers())
		return false;

	if (!streakBox.initializeBuffers())
		return false;

	if (!streakPlane.initializeBuffers())
		return false;

	if (!clipBox.initializeBuffers())
		return false;

	if (!raycasting.initializeBuffers())
		return false;

	if (!fluctuationHeightfield.initializeBuffers())
		return false;

	return true;
}


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
	ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));
	scd.BufferDesc.Width = this->windowWidth;
	scd.BufferDesc.Height = this->windowHeight;
	scd.BufferDesc.RefreshRate.Numerator = 60;
	scd.BufferDesc.RefreshRate.Denominator = 1;
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scd.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	scd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	scd.SampleDesc.Count = 4;
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
#ifdef _DEBUG //Debug Mode
		D3D11_CREATE_DEVICE_DEBUG,

#else //Release mode
		NULL,
#endif // _DEBUG or Release mode
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

	// Keep a pointer to the Adapter in SolverOptions
	solverOptions.p_Adapter = this->GetAdapter();

	return true;
}



//########################### Shader Initialization #####################
bool Graphics::InitializeShaders()
{


	if (!this->streamlineRenderer.initializeShaders())
		return false;

	if (!this->streaklineRenderer.initializeShaders())
		return false;

	if (!this->pathlineRenderer.initializeShaders())
		return false;

	if (!this->volumeBox.initializeShaders())
		return false;

	if (!this->seedBox.initializeShaders())
		return false;

	if (!this->streakBox.initializeShaders())
		return false;

	if (!this->streakPlane.initializeShaders())
		return false;

	if (!this->clipBox.initializeShaders())
		return false;

	return true;
}


// ############################# Initialize the Scene #########################

bool Graphics::InitializeScene()
{

	float center[3] = { 0,0,0 };
	DirectX::XMFLOAT4 redColor = { 1,0,0,1.0f};
	DirectX::XMFLOAT4 greenColor = { 0,1,0,1.0f };
	DirectX::XMFLOAT4 blueColor = { 0,0,1,1.0f };
	DirectX::XMFLOAT4 whiteColor = { 1,1,1,1.0f };

	volumeBox.addBox(this->solverOptions.gridDiameter, center, greenColor);
	seedBox.addBox( this->solverOptions.seedBox, this->solverOptions.seedBoxPos, redColor);

	streakBox.addBox( this->solverOptions.streakBox, this->solverOptions.streakBoxPos, redColor);

	streakPlane.addBox(this->solverOptions.streakBox, this->solverOptions.streakBoxPos, whiteColor);

	clipBox.addBox(this->raycastingOptions.clipBox, this->raycastingOptions.clipBoxCenter, blueColor);
	
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

		// initializes Options for various visualization
		this->renderImGuiOptions.setResources
		(
			&camera,&fpsTimer,
			&renderingOptions,
			&solverOptions,
			&raycastingOptions,
			&dispersionOptions,
			&spaceTimeOptions,
			&crossSectionOptions,
			&turbulentMixingOptions,
			&timeSpace3DOptions
		);


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
	this->depthStencilView.Reset();
	this->renderTargetView.Reset();

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
	this->InitializeDirectXResources();
	this->InitializeCamera();  
	this->InitializeResources();

	camera.SetPosition(-10.7f, 4.0f, -5.37f);
	camera.SetLookAtPos({ 0.75f,-0.35f,0.55f });

	if (this->renderImGuiOptions.showRaycasting)
	{
		this->raycasting.resize();
	}

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

	XMFLOAT3 viewDir = this->camera.GetViewVector();
	float3 view = { viewDir.x,viewDir.y,viewDir.z };

	return view;
}

void Graphics::saveTexture(ID3D11Texture2D* texture, std::string fileName, std::string filePath)
{
	std::string fullName = filePath + fileName;
	std::wstring wfileName = std::wstring(fullName.begin(), fullName.end());
	const wchar_t* result = wfileName.c_str();
	SaveDDSTextureToFile(this->deviceContext.Get(), texture ,result);
}

void Graphics::saveTexture(ID3D11Texture2D* texture, std::string fullName)
{
	std::wstring wfileName = std::wstring(fullName.begin(), fullName.end());
	const wchar_t* result = wfileName.c_str();
	SaveDDSTextureToFile(this->deviceContext.Get(), texture, result);
}

void Graphics::saveTextureJPEG(ID3D11Texture2D* texture, std::string fullName)
{
	std::wstring wfileName = std::wstring(fullName.begin(), fullName.end());
	const wchar_t* result = wfileName.c_str();
	SaveWICTextureToFile(this->deviceContext.Get(), texture,GUID_ContainerFormatJpeg, result);
}

