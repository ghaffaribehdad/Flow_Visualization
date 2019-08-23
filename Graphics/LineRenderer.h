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

typedef long long int llInt;


// A generic class to render lines
class LineRenderer
{

protected:
	

	// Resterizer com pointer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;

	// Buffers
	VertexBuffer<Vertex>				vertexBuffer;
	IndexBuffer							indexBuffer;
	ConstantBuffer<Tube_geometryShader> GS_constantBuffer;
	ConstantBuffer<CB_pixelShader>		PS_constantBuffer;
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

	
	
	virtual void updateIndexBuffer() {}; 			// Update Index buffer to match the vertex buffer (If Index buffer is needed)


	virtual void updateConstantBuffer(Camera& _camera);	// Update Constant buffer based on the camera positions and view 
	bool setShaders(D3D11_PRIMITIVE_TOPOLOGY Topology);					// set shaders and rasterizer
	bool initilizeRasterizer();			// Create Rasterizer state
	virtual void setBuffers();					// set vertex and index and constant buffer
	virtual bool initilizeIndexBuffer() { return true; }


public:

	virtual void addBox(float* edges, float* pos, DirectX::XMFLOAT4 color) {};			// Adds static scenes


	virtual void updateBuffers() {};						// Virutal function to implement Main Routine of the LineRenderer
	virtual void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Toplogy) {}		// Draw results to the backbuffer
	virtual bool initializeBuffers() { return true; }		// initilize vertex, constant and index buffer
	virtual void cleanPipeline();							// Deactivates the Geometry Shader prevent conflict with other pipelines
	
															// need to be called at the initilization of this object 
	//=> To Do: Move it to the constructor
	void setResources(RenderingOptions& _renderingOptions, SolverOptions& _solverOptions, ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter* pAdapter);
	
	virtual bool initializeShaders();				// Create GS,VS and PS 

};