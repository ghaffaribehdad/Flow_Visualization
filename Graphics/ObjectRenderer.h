

#pragma once

#include "LineRenderer.h"

class BoxRenderer : public LineRenderer
{

private:

	Vertex edges[24];




	void updateEdges(Camera & camera)
	{
		DirectX::XMFLOAT3 x_hat = DirectX::XMFLOAT3(1.0f, 0.0f, 0.0f);
		DirectX::XMFLOAT3 y_hat = DirectX::XMFLOAT3(0.0f, 1.0f, 0.0f);
		DirectX::XMFLOAT3 z_hat = DirectX::XMFLOAT3(0.0f, 0.0f, 1.0f);
		DirectX::XMFLOAT3 nx_hat = DirectX::XMFLOAT3(-1.0f, 0.0f, 0.0f);
		DirectX::XMFLOAT3 ny_hat = DirectX::XMFLOAT3(0.0f, -1.0f, 0.0f);
		DirectX::XMFLOAT3 nz_hat = DirectX::XMFLOAT3(0.0f, 0.0f, -1.0f);

		edges[0].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[1].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);



		edges[2].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[3].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);



		edges[4].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[5].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);



		edges[6].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[7].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);



		edges[8].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[9].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);



		edges[10].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[11].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);


		edges[12].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[13].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);

		edges[14].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[15].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);


		edges[16].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[17].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);


		edges[18].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[19].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / 2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);

		edges[20].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / -2.0f);
		edges[21].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / -2.0f);

		edges[22].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / 2.0f, this->solverOptions->gridDiameter[2] / 2.0f);
		edges[23].pos = DirectX::XMFLOAT3(this->solverOptions->gridDiameter[0] / -2.0f, this->solverOptions->gridDiameter[1] / -2.0f, this->solverOptions->gridDiameter[2] / 2.0f);


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
	}

public:
	

	void initilizeScene(Camera& camera) override
	{

		updateEdges(camera);
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


		hr = this->vertexBuffer.Initialize(this->device, edges, 24);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Vertex Buffer.");
			return false;
		}

		this->solverOptions->p_vertexBuffer = this->vertexBuffer.Get();

		return true;

	}

	void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Topology) override
	{
		initilizeRasterizer();
		setShaders(Topology);
		updateConstantBuffer(camera);
		setBuffers();
		this->deviceContext->Draw(24,0);
	}



};