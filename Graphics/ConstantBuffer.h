#pragma once

#include <d3d11.h>
#include "ConsantBufferTypes.h"
#include <wrl/client.h>
#include "..\\ErrorLogger.h"

template<class T>
class ConstantBuffer
{
private:
	// private copy constructor
	ConstantBuffer(const ConstantBuffer<T>& rhs);

private:
	Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
	ID3D11DeviceContext * deviceContext = nullptr;

public:
	ConstantBuffer() {}

	T data;

	ID3D11Buffer* Get() const
	{
		return buffer.Get();
	}

	ID3D11Buffer* const * GetAddressOf() const
	{
		return buffer.GetAddressOf();
	}


	HRESULT Initialize(ID3D11Device * device, ID3D11DeviceContext * deviceContext)
	{
		// Release if it is already alocated
		if (buffer.Get() != nullptr)
		{
			buffer.Reset();
		}
		this->deviceContext = deviceContext;


		// Create buffer description
		D3D11_BUFFER_DESC constantBufferDesc;
		ZeroMemory(&constantBufferDesc, sizeof(D3D11_BUFFER_DESC));

		constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		constantBufferDesc.MiscFlags = 0;
		constantBufferDesc.ByteWidth = static_cast<UINT>(sizeof(T) + (16 - sizeof(T) % 16)); // has to be 16-byte alligned
		constantBufferDesc.StructureByteStride = 0;

		// Create constant buffer
		HRESULT hr = device->CreateBuffer(&constantBufferDesc, 0, this->buffer.GetAddressOf());
		return hr;
	}

	bool ApplyChanges()
	{
		D3D11_MAPPED_SUBRESOURCE  mappedResource;
		HRESULT hr = this->deviceContext->Map(buffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to map constant buffer.");
			return false;
		}
		CopyMemory(mappedResource.pData, &data, sizeof(T));
		this->deviceContext->Unmap(buffer.Get(), 0);
		this->deviceContext->Unmap(buffer.Get(), 0);
		
		return true;
	}
};