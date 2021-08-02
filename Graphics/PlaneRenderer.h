

#pragma once

#include "LineRenderer.h"



class PlaneRenderer : public LineRenderer
{
public:
	virtual void show(RenderImGuiOptions* renderImGuiOptions, float * diamter, float * center = nullptr)
	{
		if (center == nullptr)
		{
			float origin[3] = { 0.0f,0.0f,0.0f };
			this->updateScene(diamter, origin);
		}
		else
		{
			*center = *center + 0.01f;
			this->updateScene(diamter, center);
		}

		renderImGuiOptions->updateVolumeBox = false;
	}

	virtual bool release() override
	{
		return true;
	}


private:

	Vertex edges[6];
	DirectX::XMFLOAT4 color;


	void updateEdges(float * p_gridDiameter, float * gridPos)
	{
		float gridDiameter[3] = { 0.0f,0.0f,0.0f };

		for (int i = 0; i < 3; i++)
		{
			gridDiameter[i] = p_gridDiameter[i];
		}


		edges[0].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] - gridDiameter[1] / 2.0f , gridPos[2] - gridDiameter[2] / 2.0f);	// -, -
		edges[1].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] + gridDiameter[1] / 2.0f , gridPos[2] - gridDiameter[2] / 2.0f);	// +, -
		edges[2].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] + gridDiameter[1] / 2.0f , gridPos[2] + gridDiameter[2] / 2.0f);	// +, +

		edges[3].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] - gridDiameter[1] / 2.0f, gridPos[2] - gridDiameter[2] / 2.0f);		// -, -
		edges[4].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] + gridDiameter[1] / 2.0f, gridPos[2] + gridDiameter[2] / 2.0f);		// +, +
		edges[5].pos = DirectX::XMFLOAT3(gridPos[0], gridPos[1] - gridDiameter[1] / 2.0f, gridPos[2] + gridDiameter[2] / 2.0f);		// -, +
	}


	bool initilizeIndexBuffer() override
	{

		// A Box has 6 vertices each contains two points

		indices.resize(6);

		HRESULT hr = this->indexBuffer.Initialize(this->device, &indices.at(0), indices.size());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Index Buffer.");
			return false;
		}

		return true;
	}




	void setBuffers() override
	{

		UINT offset = 0;

		this->deviceContext->IASetIndexBuffer(indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); // set Vertex buffer
		this->deviceContext->GSSetConstantBuffers(0, 1, this->GS_constantBuffer.GetAddressOf());
		this->deviceContext->PSSetConstantBuffers(0, 1, this->PS_constantBuffer.GetAddressOf());
	}

	void updateVertexBuffer()
	{
		this->vertexBuffer.Get()->Release();


	}

public:




	bool updateScene(float * _edges, float * pos)
	{
		this->updateEdges(_edges, pos);
		HRESULT hr = this->vertexBuffer.Initialize(this->device, this->edges, 6);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		UINT offset = 0;

		this->deviceContext->IASetVertexBuffers(0, 1, this->vertexBuffer.GetAddressOf(), this->vertexBuffer.StridePtr(), &offset); // set Vertex buffer

		return true;
	}

	void addBox(float * edges, float * pos, DirectX::XMFLOAT4 _color) override
	{

		updateEdges(edges, pos);
		this->color = _color;
	}

	bool initializeShaders() override
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

		if (!this->geometryshader.Initialize(this->device, shaderfolder + L"geometryshaderplane.cso"))
		{
			return false;
		}

		if (!this->pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderPlane.cso"))
		{
			return false;
		}

		return true;
	}


	bool initializeBuffers() override
	{

		HRESULT hr = this->GS_constantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Constant buffer.");
			return false;
		}


		hr = this->vertexBuffer.Initialize(this->device, this->edges, 6);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		hr = this->PS_constantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Pixel shader Constant buffer.");
			return false;
		}


		return true;

	}



	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
		initializeRasterizer();

		this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
		this->deviceContext->IASetPrimitiveTopology(Topology);							// set the primitive topology
		this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
		this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
		this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);
		this->deviceContext->GSSetShader(geometryshader.GetShader(), NULL, 0);
		this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);

		updateConstantBuffer(camera);
		setBuffers();
		this->deviceContext->Draw(6, 0);
		this->cleanPipeline();
	}


	void updateConstantBuffer(Camera & camera) override
	{

		DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

		// Set attributes of constant buffer for geometry shader
		GS_constantBuffer.data.View = world * camera.GetViewMatrix();
		GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
		GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
		GS_constantBuffer.data.tubeRadius = renderingOptions->boxRadius;
		GS_constantBuffer.data.viewDir = camera.GetViewVector();

		PS_constantBuffer.data.minColor = color;

		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}
};