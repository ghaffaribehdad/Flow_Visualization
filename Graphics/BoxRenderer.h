

#pragma once

#include "LineRenderer.h"



class BoxRenderer : public LineRenderer
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
			this->updateScene(diamter, center);
		}
		
		renderImGuiOptions->updateVolumeBox = false;
	}

	virtual bool release() override
	{
		return true;
	}


private:

	Vertex edges[24];
	DirectX::XMFLOAT4 color;


	void updateEdges( float * p_gridDiameter, float * gridPos)
	{
		float gridDiameter[3] = { 0.0f,0.0f,0.0f };

		for (int i = 0; i < 3; i++)
		{
			gridDiameter[i] = p_gridDiameter[i] ;
		}


		edges[0].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[1].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);



		edges[2].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[3].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);



		edges[4].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[5].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);



		edges[6].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[7].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);



		edges[8].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[9].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);



		edges[10].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[11].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);


		edges[12].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[13].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);

		edges[14].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[15].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);


		edges[16].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[17].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);


		edges[18].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[19].pos = DirectX::XMFLOAT3(gridDiameter[0] / 2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);

		edges[20].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);
		edges[21].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / -2.0f +  gridPos[2]);

		edges[22].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / 2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);
		edges[23].pos = DirectX::XMFLOAT3(gridDiameter[0] / -2.0f +  gridPos[0], gridDiameter[1] / -2.0f +  gridPos[1], gridDiameter[2] / 2.0f +  gridPos[2]);


		for (int i = 0; i < 8; i++)
		{
			edges[2*i].LineID = 0;
			edges[2 * i + 1].LineID = 0;

			edges[2*i].measure = 0;
			edges[2*i+1].measure = 0;

			XMVECTOR pos1 = XMLoadFloat3(&edges[2 * i + 1].pos);
			XMVECTOR pos0 = XMLoadFloat3(&edges[2 * i].pos);

			XMVECTOR tan = XMVector3Normalize(pos1 - pos0);

			XMStoreFloat3(&edges[2 * i].tangent, tan);
			XMStoreFloat3(&edges[2 * i+1].tangent, tan);

		}
		for (int i = 8; i < 12; i++)
		{
			edges[2 * i].LineID = 1;
			edges[2 * i + 1].LineID = 1;



			edges[2 * i].measure = 0;
			edges[2 * i + 1].measure = 0;

			XMVECTOR pos1 = XMLoadFloat3(&edges[2 * i + 1].pos);
			XMVECTOR pos0 = XMLoadFloat3(&edges[2 * i].pos);

			XMVECTOR tan = XMVector3Normalize(pos1 - pos0);

			XMStoreFloat3(&edges[2 * i].tangent, tan);
			XMStoreFloat3(&edges[2 * i + 1].tangent, tan);

		}
	}


	bool initilizeIndexBuffer() override
	{

		// A Box has 12 vertices each contains two points

		indices.resize(24);

		HRESULT hr = this->indexBuffer.Initialize(this->device, &indices.at(0),  indices.size());
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
		vertexBuffer.reset();
		HRESULT hr = device->GetDeviceRemovedReason();
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "why??");
			return false;
		}


		 hr = this->vertexBuffer.Initialize(this->device, this->edges, 24);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer Box Renderer.");
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

		if (!this->geometryshader.Initialize(this->device, shaderfolder + L"geometryshaderLineTubeBox.cso"))
		{
			return false;
		}

		if (!this->pixelshader.Initialize(this->device, shaderfolder + L"pixelshaderSeedBox.cso"))
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
		vertexBuffer.reset();

		hr = this->vertexBuffer.Initialize(this->device, this->edges, 24);
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
		setShaders(Topology);
		if (Topology == D3D11_PRIMITIVE_TOPOLOGY::D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
		{
			this->deviceContext->GSSetShader(NULL, NULL, 0);
		}

		updateConstantBuffer(camera);
		setBuffers();
		this->deviceContext->Draw(24,0);
		this->cleanPipeline();
	}


	void updateConstantBuffer(Camera & camera) override
	{

		DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

		// Set attributes of constant buffer for geometry shader
		GS_constantBuffer.data.View = world * camera.GetViewMatrix();
		GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
		GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
		GS_constantBuffer.data.tubeRadius =renderingOptions->boxRadius;
		GS_constantBuffer.data.viewDir = camera.GetViewVector();

		PS_constantBuffer.data.minColor = color;

		// Update Constant Buffer
		GS_constantBuffer.ApplyChanges();
		PS_constantBuffer.ApplyChanges();
	}
};