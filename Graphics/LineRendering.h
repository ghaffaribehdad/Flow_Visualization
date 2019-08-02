#pragma once

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "constantBuffer.h"
#include "Shaders.h"
#include "Vertex.h"
#include "ConsantBufferTypes.h"
#include <d3d11.h>


class lineRendering
{

protected:

	VertexBuffer<Vertex> vertexBuffer;
	VertexShader vShader;
	PixelShader pShader;
	IndexBuffer indexBuffer;

	Microsoft::WRL::ComPtr<ID3D11DeviceContext>	deviceContext;

	void initializeVertexBuffer();
	void initializeIndexBuffer();
	void initializeVertexShader();
	void initializePixelShader();

public:

	void initialize();
	void draw();

};