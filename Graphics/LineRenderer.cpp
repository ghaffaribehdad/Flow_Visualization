#include "LineRenderer.h"



bool LineRenderer::setShaders(D3D11_PRIMITIVE_TOPOLOGY Topology)
{

	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(Topology);							// set the primitive topology
	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);		
	this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);
	//this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);


	return true;
}




void LineRenderer::setBuffers()
{
	
	UINT offset = 0;

	//set index and vertex buffer
	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); // set Vertex buffer
	this->deviceContext->GSSetConstantBuffers(0, 1, this->GS_constantBuffer.GetAddressOf());
	this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBuffer.GetAddressOf());
}



void LineRenderer::updateConstantBuffer(Camera& camera)
{


	DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

	// Set attributes of constant buffer for geometry shader
	GS_constantBuffer.data.View = world * camera.GetViewMatrix();
	GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
	GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
	GS_constantBuffer.data.tubeRadius = renderingOptions->tubeRadius;
	GS_constantBuffer.data.viewDir = camera.GetViewVector();


	// Update Constant Buffer
	GS_constantBuffer.ApplyChanges();

}



void LineRenderer::setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions,ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter * _adapter)
{
	this->solverOptions = &_solverOptions;
	this->solverOptions->p_Adapter = _adapter;
	this->renderingOptions = &_renderingOptions;
	this->device = _device;
	this->deviceContext = _deviceContext;
}
bool LineRenderer::initializeShaders()
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
		{"LINEID",0,DXGI_FORMAT::DXGI_FORMAT_R32_UINT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"MEASURE",0,DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"NORMAL",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"INITIALPOS",0,DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0}
	};
	UINT numElements = ARRAYSIZE(layout);

	if (!this->vertexshader.Initialize(this->device, shaderfolder + L"vertexshader.cso", layout, numElements))
	{
		return false;
	}


	switch (renderingOptions->renderingMode)
	{
	case RenderingMode::RenderingMode::TUBES:
	{
		if (!this->geometryshader.Initialize(this->device, shaderfolder + L"geometryshaderLineTube.cso"))
		{
			return false;
		}

		if (!this->pixelshader.Initialize(this->device, shaderfolder + L"pixelshader.cso"))
		{
			return false;
		}
		break;
	}
	case RenderingMode::RenderingMode::SPHERES:
	{

		if (!this->geometryshader.Initialize(this->device, shaderfolder + L"geometryshaderSphere.cso"))
		{
			return false;
		}



		if (!this->pixelshader.Initialize(this->device, shaderfolder + L"pixelShaderSphere.cso"))
		{
			return false;
		}
		break;
	}
	}




	return true;
}




void LineRenderer::cleanPipeline()
{
	this->deviceContext->GSSetShader(NULL, NULL, 0);
}

bool LineRenderer::releaseScene()
{
	this->vertexBuffer.Get()->Release();
	return true;
}



bool LineRenderer::initializeRasterizer()
{
	if (this->rasterizerstate.Get() == nullptr)
	{
		// Create Rasterizer state
		D3D11_RASTERIZER_DESC rasterizerDesc;
		ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE; // CULLING could be set to none
		rasterizerDesc.MultisampleEnable = true;
		rasterizerDesc.AntialiasedLineEnable = true;
		//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;






		HRESULT hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
			return false;
		}



		//Create the blend state
		D3D11_BLEND_DESC blendDesc;
		ZeroMemory(&blendDesc, sizeof(blendDesc));

		D3D11_RENDER_TARGET_BLEND_DESC rtbd;
		ZeroMemory(&rtbd, sizeof(rtbd));

		rtbd.BlendEnable = true;

		rtbd.SrcBlend = D3D11_BLEND::D3D11_BLEND_SRC_ALPHA;
		rtbd.SrcBlendAlpha = D3D11_BLEND::D3D11_BLEND_ONE;

		rtbd.DestBlend = D3D11_BLEND::D3D11_BLEND_INV_SRC_ALPHA;
		rtbd.DestBlendAlpha = D3D11_BLEND::D3D11_BLEND_INV_SRC_ALPHA;

		rtbd.BlendOp = D3D11_BLEND_OP::D3D11_BLEND_OP_ADD;
		rtbd.BlendOpAlpha = D3D11_BLEND_OP::D3D11_BLEND_OP_ADD;

		rtbd.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE::D3D11_COLOR_WRITE_ENABLE_ALL;


		blendDesc.RenderTarget[0] = rtbd;

		hr = this->device->CreateBlendState(&blendDesc, this->blendState.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to create blend state.");
			return false;
		}
	}

	return true;
}