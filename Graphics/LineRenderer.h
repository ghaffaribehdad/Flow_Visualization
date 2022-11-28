#pragma once

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "constantBuffer.h"
#include "Shaders.h"
#include "Vertex.h"
#include "ConsantBufferTypes.h"
#include <d3d11.h>
#include "../Options/RenderingOptions.h"
#include "../Options/SolverOptions.h"
#include "../Options/FieldOptions.h"
#include <Windows.h>
#include "Camera.h"
#include "../Graphics/RenderImGuiOptions.h"

typedef long long int llInt;


// A generic class to render lines
class LineRenderer
{

public:
	ID3D11RenderTargetView*				mainRTV;
	ID3D11DepthStencilView*				depthstencil;
protected:
	
	int counter = 0;
	bool updateOIT = false;
	// Resterizer com pointer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;
	Microsoft::WRL::ComPtr<ID3D11BlendState>			blendState;

	// Buffers
	VertexBuffer<Vertex>				vertexBuffer;
	IndexBuffer							indexBuffer;
	ConstantBuffer<Tube_geometryShader> GS_constantBuffer;
	ConstantBuffer<CB_VS_Sampler>		VS_SamplerConstantBuffer;
	ConstantBuffer<CB_pixelShader>		PS_constantBuffer;
	ConstantBuffer<CB_pixelShaderSampler> PS_constantBufferSampler;
	std::vector<DWORD>					indices;






	// Viewport dimension
	int width = 0;
	int height = 0;

	// Shaders
	VertexShader		vertexshader;
	VertexShader		vertexshaderSecondPass;
	VertexShader		vertexshaderSampler;

	PixelShader			pixelshader;
	PixelShader			pixelshaderFirstPass;
	PixelShader			pixelshaderSecondPass;
	PixelShader			pixelShaderSampler;
	GeometryShader		geometryshader;



	// Reference of resources
	RenderingOptions	* renderingOptions;
	SolverOptions		* solverOptions;
	FieldOptions		* fieldOptions;



	// Pointers to graphics infrastructures
	ID3D11DeviceContext*	deviceContext;
	ID3D11Device*			device;
	IDXGIAdapter*			pAdapter = nullptr;


	
	
	virtual void updateIndexBuffer() {}; 			// Update Index buffer to match the vertex buffer (If Index buffer is needed)




	virtual void updateConstantBuffer(Camera& _camera);	// Update Constant buffer based on the camera positions and view 
	virtual bool setShaders(D3D11_PRIMITIVE_TOPOLOGY Topology);					// set shaders and rasterizer
	bool initializeRasterizer();						// Create Rasterizer state
	virtual void setBuffers();							// set vertex and index and constant buffer
	virtual bool initilizeIndexBuffer() { return true; }
	virtual void resetRealtime() {};
public:

	virtual void addBox(float* edges, float* pos, DirectX::XMFLOAT4 color) {};			// Adds static scenes
	virtual bool release() = 0;
	virtual void updateBuffers() {};						// Virutal function to implement Main Routine of the LineRenderer
	virtual bool updateDraw() { return true; };
	virtual void draw(Camera& camera, D3D11_PRIMITIVE_TOPOLOGY Toplogy) {}		// Draw results to the backbuffer
	virtual bool initializeBuffers() { return true; }		// initilize vertex, constant and index buffer
	virtual void cleanPipeline();							// Deactivates the Geometry Shader prevent conflict with other pipelines
	bool releaseScene();
	float streakProjectionPlane();
	float streakProjectionPlaneFix();
	float streakProjectionPlane_Stream();
															// need to be called at the initilization of this object 
	//=> To Do: Move it to the constructor
	virtual void setResources(RenderingOptions* _renderingOptions, SolverOptions* _solverOptions, FieldOptions* _fieldOptions, ID3D11DeviceContext* _deviceContext, ID3D11Device* _device, IDXGIAdapter* pAdapter, const int & width = 0, const int & height = 0);
	
	virtual bool initializeShaders();				// Create GS,VS and PS 

};