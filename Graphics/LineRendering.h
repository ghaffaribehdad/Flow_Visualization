#pragma once

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "constantBuffer.h"
#include "Shaders.h"
#include "Vertex.h"
#include "ConsantBufferTypes.h"
#include <d3d11.h>
#include "RenderingOptions.h"
#include <Windows.h>
#include "Camera.h"


class LineRendering
{

protected:


	// Resterizer com pointer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;

	// Buffers
	VertexBuffer<Vertex>				vertexBuffer;
	IndexBuffer							indexBuffer;
	ConstantBuffer<Tube_geometryShader> GS_constantBuffer;


	// Shaders
	VertexShader		vertexshader;
	PixelShader			pixelshader;
	GeometryShader		geometryshader;

	// Rendering Options
	RenderingOptions	renderingOptions;


	// Pointers to graphics infrastructures
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>	deviceContext;
	Microsoft::WRL::ComPtr<ID3D11Device>		device;

	

	bool initializeShaders();
	bool initializeConstantBuffer(Camera & camera);
	bool initializeIndexBuffer();

	bool updateConstantBuffer();
	bool updateIndexBuffer();
	bool updateVertexBuffer();

	bool setShaders();
	bool setBuffers();
	
	// After drawing the pipeline must be clean (at least geometry shader needs to be deactivated)
	bool cleanPipeline();


public:

	void initialize();
	void draw();
	void update();
	
};