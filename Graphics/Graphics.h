#pragma once

#include "AdapterReader.h"
#include "Shaders.h"
#include "Vertex.h"

#include <WICTextureLoader.h>
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "ConsantBufferTypes.h"
#include "ConstantBuffer.h"
#include "Camera.h"
#include "..\Timer.h"
#include "..\Cuda\CudaSolver.h"
#include "ImGui/imgui.h"

#include <d3d11.h>
#include "..\\Raycaster\Raycasting.h"

#include "RenderingOptions.h"
#include "StreamlineRenderer.h"
#include "BoxRenderer.h"
#include "RenderImGuiOptions.h"
#include "PathlineRenderer.h"


typedef long long int llInt;


class Graphics
{
	friend CUDASolver;

public:
	// Initialize graphics
	bool Initialize(HWND hwnd, int width, int height);

	// main rendering function
	void RenderFrame();

	//  Resize the window
	void Resize(HWND hwnd);

	// Add Camera object
	Camera camera;

	SolverOptions solverOptions;
	RenderingOptions renderingOptions;
	RaycastingOptions raycastingOptions;

	// Getter Functions
	IDXGIAdapter* GetAdapter();
	ID3D11Device* GetDevice();
	ID3D11DeviceContext* GetDeviceContext()
	{
		return this->deviceContext.Get();
	}

	ID3D11Buffer* GetVertexBuffer();




	// Get the camera position and directions
	const float3 getUpVector();
	const float3 getEyePosition();
	const float3 getViewDir();
	const int& getWindowHeight()
	{
		return this->windowHeight;
	}
	const int& getWindowWidth()
	{
		return this->windowWidth;
	}

	const float& getFOV()
	{
		return this->FOV;
	}



	// for now it is public while imgui need the access
	Timer fpsTimer;

private:

	// camera propertis
	float FOV = 30.0;


	// call by Initialize() funcion
	bool InitializeDirectX(HWND hwnd);
	bool InitializeDirectXResources();
	bool InitializeResources();
	bool InitializeShaders();
	bool InitializeScene();
	bool InitializeCamera();


	// ImGui
	bool InitializeImGui(HWND hwnd);
	ImGuiContext* ImGuicontext = nullptr;


	ID3D11Texture2D* getBackBuffer()
	{
		Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
		HRESULT hr = this->swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backBuffer.GetAddressOf()));
		if (FAILED(hr)) //If error occurred
		{
			ErrorLogger::Log(hr, "GetBuffer Failed.");
			return nullptr;
		}

		return backBuffer.Get();

	}


	// directx resources
	Microsoft::WRL::ComPtr<ID3D11Device>			device;// use to creat buffers
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>		deviceContext; //use to set resources for rendering
	Microsoft::WRL::ComPtr<IDXGISwapChain>			swapchain; // use to swap out our frame
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	renderTargetView; // where we are going to render our buffer
	Microsoft::WRL::ComPtr<ID3D11SamplerState>		samplerState;	// For depth test between raycasting and line rendering


	// Shaders
	VertexShader vertexshader;
	PixelShader pixelshader;
	GeometryShader geometryshader;


	// Buffers
	VertexBuffer<Vertex> vertexBuffer;
	IndexBuffer indexBuffer;


	// Depth stencil view and buffer and state
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView>		depthStencilView;
	Microsoft::WRL::ComPtr<ID3D11Texture2D>				depthStencilBuffer;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState>		depthStencilState;



	// Pointer to the adapter
	IDXGIAdapter* adapter;

	int windowWidth = 0;
	int windowHeight = 0;

	// Rendering Obejcts
	PathlineRenderer pathlineRenderer;
	StreamlineRenderer streamlineRenderer;
	BoxRenderer volumeBox;
	BoxRenderer seedBox;

	// Raycasting (This object would write into a texture and pass it to the graphics then we need to use sampler state to show it on the backbuffer)
	Raycasting raycasting;


	// Solver options
	bool streamline = true;
	bool pathline = false;

	void saveTexture(ID3D11Texture2D* texture);

	Vertex* CudaVertex = nullptr;


	RenderImGuiOptions renderImGuiOptions;

};