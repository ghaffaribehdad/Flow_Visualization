#include "LineRenderer.h"

typedef long long int llInt;


bool LineRenderer::setShaders()
{

	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP);// set the primitive topology
	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);		
	this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);


	return true;
}


bool LineRenderer::initializeBuffers()
{
	this->indices.resize(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count));
	
	HRESULT hr = this->GS_constantBuffer.Initialize(this->device, this->deviceContext);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Constant buffer.");
		return false;
	}

	hr = this->indexBuffer.Initialize(this->device, &indices.at(0), indices.size());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
		return false;
	}

	hr = this->vertexBuffer.Initialize(this->device, NULL, solverOptions->lineLength * solverOptions->lines_count);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
		return false;
	}



	return true;

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
		D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA,0}
	};
	UINT numElements = ARRAYSIZE(layout);

	if (!this->vertexshader.Initialize(this->device, shaderfolder + L"vertexshader.cso", layout, numElements))
	{
		return false;
	}

	if (!this->geometryshader.Initialize(this->device, shaderfolder + L"geometryshader.cso"))
	{
		return false;
	}

	if (!this->pixelshader.Initialize(this->device, shaderfolder + L"pixelshader.cso"))
	{
		return false;
	}

	return true;
}


void LineRenderer::setBuffers()
{
	
	UINT offset = 0;

	//set index and vertex buffer
	this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); // set Vertex buffer
	this->deviceContext->IASetIndexBuffer(this->indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0); // Set index buffer
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




void LineRenderer::updateIndexBuffer()
{

	indices.resize(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count));
	for (int i = 0; i < indices.size(); i++)
	{

		indices[i] = i;

	}

}

bool LineRenderer::initialize()
{
	
	if (!this->initializeBuffers())
		return false;

	if (!this->initilizeRasterizer())
		return false;

	if (!this->initializeShaders())
		return false;

	return true;
}

void LineRenderer::setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions,ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter * _adapter)
{
	this->solverOptions = &_solverOptions;
	this->solverOptions->p_Adapter = _adapter;
	this->renderingOptions = &_renderingOptions;
	this->device = _device;
	this->deviceContext = _deviceContext;
}

void LineRenderer::draw(Camera & camera)
{	
	
	this->setShaders();
	this->updateView(camera);
	this->updateIndexBuffer();
	this->setBuffers();
	this->deviceContext->DrawIndexed(llInt(solverOptions->lineLength) * llInt(solverOptions->lines_count), 0,0);
}

void LineRenderer::cleanPipeline()
{
	this->deviceContext->GSSetShader(NULL, NULL, 0);
}

void LineRenderer::updateScene()
{
	
	this->updateIndexBuffer();
}

void LineRenderer::updateView(Camera& camera)
{
	this->updateConstantBuffer(camera);
}


bool LineRenderer::initilizeRasterizer()
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
	}
}