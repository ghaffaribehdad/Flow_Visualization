#pragma once
#include "Graphics.h"
// #include <ScreenGrab11.h>
#include <dxgidebug.h>
#include <dxgi1_3.h>
#include <wincodec.h>


bool createTexture(
	uint width,
	uint height,
	ID3D11Device * device,
	ID3D11Texture2D ** texture,
	D3D11_BIND_FLAG bindFlag = D3D11_BIND_RENDER_TARGET,
	DXGI_SAMPLE_DESC sample_desc = { 4,0 },
	DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
	uint mipLevel = 1
	)
{
	D3D11_TEXTURE2D_DESC            TexDesc;
	// Create a screen-sized depth stencil resource
	// Use a full 32-bits format for depth when depth peeling is used
	// This is to avoid Z-fighting artefacts due to the "manual" depth buffer implementation
	TexDesc.Width = width;
	TexDesc.Height = height;
	TexDesc.Format = format;
	TexDesc.SampleDesc = sample_desc;
	TexDesc.MipLevels = mipLevel;
	TexDesc.Usage = D3D11_USAGE_DEFAULT;
	TexDesc.MiscFlags = 0;
	TexDesc.CPUAccessFlags = 0;
	TexDesc.BindFlags = bindFlag;
	TexDesc.ArraySize = 1;
	HRESULT hr = device->CreateTexture2D(&TexDesc, NULL, texture);
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Texture");
		return false;
	}

	return true;
}
