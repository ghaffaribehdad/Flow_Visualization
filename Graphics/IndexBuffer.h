#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <vector>

class IndexBuffer
{
private:
	IndexBuffer(const IndexBuffer & rhs);

private:
	Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
	UINT buffersize = 0;

public:
	IndexBuffer() {}

	UINT GetBuffersize() const
	{
		return this->buffersize;
	}
	
	ID3D11Buffer * Get() const
	{
		return this->buffer.Get();
	}

	ID3D11Buffer * const * GetAddressOf() const
	{
		return this->buffer.GetAddressOf();
	}

	HRESULT Initialize(ID3D11Device * device, DWORD * data, UINT numIndices)
	{
		// Release if it is already alocated
		if (buffer.Get() != nullptr)
		{
			buffer.Reset();
		}
		this->buffersize = numIndices;
		
		// Create Description structure for Indices Buffer
		D3D11_BUFFER_DESC indexBufferDesc;
		ZeroMemory(&indexBufferDesc, sizeof(indexBufferDesc));
		indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
		indexBufferDesc.ByteWidth = sizeof(DWORD)*numIndices;
		indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		indexBufferDesc.CPUAccessFlags = 0;
		indexBufferDesc.MiscFlags = 0;

		// Create Indices Buffer Data Structure
		D3D11_SUBRESOURCE_DATA indexBufferData;
		indexBufferData.pSysMem = data;

		// Create Indices Buffer
		HRESULT hr = device->CreateBuffer(&indexBufferDesc, &indexBufferData, this->buffer.GetAddressOf());

		return hr;
	}

};