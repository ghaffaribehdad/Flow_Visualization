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
	this->deviceContext->OMSetBlendState(NULL, NULL, 0xFFFFFFFF);


	
	return true;
}




void LineRenderer::setBuffers()
{
	
	UINT offset = 0;
	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset);	// Vertex buffer
	this->deviceContext->GSSetConstantBuffers(0, 1, this->GS_constantBuffer.GetAddressOf());									// Geometry shader constant buffer
	this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBuffer.GetAddressOf());									// Pixel shader constant buffer

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





void LineRenderer::setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions,ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter * _adapter , const int & _width, const int & _height, ID3D11Texture2D* _mainRT)
{
	this->solverOptions = &_solverOptions;
	this->solverOptions->p_Adapter = _adapter;
	this->renderingOptions = &_renderingOptions;
	this->device = _device;
	this->deviceContext = _deviceContext;
	this->width = _width;
	this->height = _height;
	this->pRT = _mainRT;
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
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0},
		{"TIME",0,DXGI_FORMAT::DXGI_FORMAT_R32_UINT,0,D3D11_APPEND_ALIGNED_ELEMENT,
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0}
	};
	UINT numElements = ARRAYSIZE(layout);

	if (!this->vertexshader.Initialize(this->device, shaderfolder + L"vertexshader.cso", layout, numElements))
	{
		return false;
	}


	D3D11_INPUT_ELEMENT_DESC layoutSecondPass[] =
	{
		{
			"POSITION",
			0,
			DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,
			0,
			D3D11_APPEND_ALIGNED_ELEMENT,
			D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,
			0
		},

		{
			"TEXCOORD",
			0,
			DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT,
			0,
			D3D11_APPEND_ALIGNED_ELEMENT,
			D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,
			0
		}
	};



	numElements = ARRAYSIZE(layoutSecondPass);
	if (!vertexshaderSecondPass.Initialize(this->device, shaderfolder + L"vertexShaderSecondPass.cso", layoutSecondPass, numElements))
	{
		return false;

	}

	numElements = ARRAYSIZE(layoutSecondPass);
	if (!vertexshaderSampler.Initialize(this->device, shaderfolder + L"vertexShaderSampler.cso", layoutSecondPass, numElements))
	{
		return false;

	}

	if (!this->pixelshaderSecondPass.Initialize(this->device, shaderfolder + L"secondPassPS.cso"))
	{
		return false;
	}

	if (!this->pixelShaderSampler.Initialize(this->device, shaderfolder + L"pixelshaderSampler.cso"))
		return false;


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

		if (!this->pixelshaderFirstPass.Initialize(this->device, shaderfolder + L"firstPassPS.cso"))
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

float LineRenderer::streakProjectionPlane()
{
	int current		= solverOptions->currentIdx;
	float timeDim	= solverOptions->timeDim;
	int lastIdx		= solverOptions->lastIdx;
	int	firstIdx	= solverOptions->firstIdx;
	
	float init_pos =  - timeDim / 2;
	init_pos += (current - firstIdx) * (timeDim / (lastIdx -firstIdx +1));


	return init_pos;
}

float LineRenderer::streakProjectionPlane_Stream()
{
	int current = solverOptions->currentSegment;
	float timeDim = solverOptions->timeDim;
	int lastIdx = solverOptions->lineLength;
	int	firstIdx = 0;

	float init_pos = -timeDim / 2;
	init_pos += (current - firstIdx) * (timeDim / (lastIdx - firstIdx + 1));


	return init_pos;
}


bool LineRenderer::initializeRasterizer()
{
	
	// Create Rasterizer state
	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

	rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE; // CULLING could be set to none
	rasterizerDesc.MultisampleEnable = TRUE;
	rasterizerDesc.AntialiasedLineEnable = TRUE;
	HRESULT hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
		return false;
	}

	return true;
}