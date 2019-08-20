#pragma once

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "constantBuffer.h"
#include "Shaders.h"
#include "Vertex.h"
#include "ConsantBufferTypes.h"
#include <d3d11.h>
#include "RenderingOptions.h"
#include "..//SolverOptions.h"
#include <Windows.h>
#include "Camera.h"


class LineRenderer
{

public:


	// Resterizer com pointer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;

	// Buffers
	VertexBuffer<Vertex>				vertexBuffer;
	IndexBuffer							indexBuffer;
	ConstantBuffer<Tube_geometryShader> GS_constantBuffer;
	std::vector<DWORD>					indices;


	// Shaders
	VertexShader		vertexshader;
	PixelShader			pixelshader;
	GeometryShader		geometryshader;

	// Reference of resources
	RenderingOptions	* renderingOptions;
	SolverOptions		* solverOptions;



	// Pointers to graphics infrastructures
	ID3D11DeviceContext*	deviceContext;
	ID3D11Device*			device;
	IDXGIAdapter*			pAdapter = nullptr;

	
	





public:
	// initilize GS,VS and PS 
	bool initializeShaders();

	// initilize vertex, constant and index buffer
	bool initializeBuffers();

	bool initilizeRasterizer();

	// Update Constant Buffer (view + tube radius)
	void updateConstantBuffer(Camera& camera);

	// Update Index buffer to match the vertex buffer
	void updateIndexBuffer();


	// set shaders and rasterizer
	bool setShaders();

	// set vertex and index and constant bufferbuffer
	void setBuffers();

	// After drawing the pipeline must be clean (at least geometry shader needs to be deactivated)

	void updateView(Camera& camera);
	void setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions, ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter* pAdapter);
	void cleanPipeline();
	bool initialize();
	void draw(Camera& camera);

	virtual void updateScene();
	
};