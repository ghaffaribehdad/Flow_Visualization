#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>

template<class T>
class VertexBuffer
{
private:
	// private copy constructor
	VertexBuffer(const VertexBuffer<T>& rhs);

private:
	Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
	std::unique_ptr<UINT> stride;
	UINT bufferSize = 0;

public:
	VertexBuffer(){}
	
	ID3D11Buffer* Get() const
	{
		return buffer.Get();
	}

	ID3D11Buffer* const * GetAddressOf() const
	{
		return buffer.GetAddressOf();
	}

	UINT BufferSize() const
	{
		return this->bufferSize;
	}

	UINT Stride() const
	{
		return *this->stride.get();
	}

	UINT * StridePtr() const
	{
		return this->stride.get();
	}

	HRESULT Initialize(ID3D11Device * device, T * data, UINT numVertices, unsigned int accessFlag = 0 )
	{
		// Release if it is already alocated
		if (buffer.Get() != nullptr)
		{
			buffer.Reset();
		}
		this->bufferSize = numVertices;
		if (this->stride.get() == nullptr)
		{
			this->stride = std::make_unique<UINT>(sizeof(T));
		}

		// Create buffer description
		D3D11_BUFFER_DESC vertexBufferDesc;
		ZeroMemory(&vertexBufferDesc, sizeof(D3D11_BUFFER_DESC));

		vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		vertexBufferDesc.ByteWidth = sizeof(T)*numVertices;
		vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		vertexBufferDesc.CPUAccessFlags = 0;
		vertexBufferDesc.MiscFlags = accessFlag;
	

		// Create Vertex buffer Data structure
		if (data != NULL)
		{
			D3D11_SUBRESOURCE_DATA vertexBufferData;
			ZeroMemory(&vertexBufferData, sizeof(vertexBufferData));
			vertexBufferData.pSysMem = data;

			// Create Vertex Buffer
			HRESULT hr = device->CreateBuffer(&vertexBufferDesc, &vertexBufferData, this->buffer.GetAddressOf());
			return hr;
		}
		else
		{
			HRESULT hr = device->CreateBuffer(&vertexBufferDesc, NULL, this->buffer.GetAddressOf());
			return hr;
		}

	}
};