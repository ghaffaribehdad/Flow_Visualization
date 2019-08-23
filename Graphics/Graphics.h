#pragma once

#include "AdapterReader.h"
#include "Shaders.h"
#include "Vertex.h"
#include <SpriteBatch.h>
#include <SpriteFont.h>
#include <WICTextureLoader.h>
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "ConsantBufferTypes.h"
#include "ConstantBuffer.h"
#include "Camera.h"
#include "..\Timer.h"
#include "..\Cuda\CudaSolver.h"
#include "ImGui/imgui.h"

#include "..\\Cuda\Interoperability.cuh"
#include <d3d11.h>
#include "..\\Cuda\cudaSurface.cuh"
#include "..\\testCudaInterOp.cuh"
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

	// Getter Functions
	IDXGIAdapter* GetAdapter();
	ID3D11Device* GetDevice();
	ID3D11DeviceContext* GetDeviceContext()
	{
		return this->deviceContext.Get();
	}
	ID3D11Texture2D* getBackbuffer()
	{
		HRESULT hr = this->swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(frontTex.GetAddressOf()));
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Get Back buffer");
		}
		return this->frontTex.Get();
	}

	ID3D11Buffer* GetVertexBuffer();


	bool showLines = false;


	// Check comments inside the definition
	bool initializeRaycasting();

	bool releaseRaycastingResource()
	{
		// destroy and release the resources
		cudaSurface.destroySurface();
		cudaRayTracingInteroperability.release();
	}


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

	CudaSurface* getSurfaceObject()
	{
		return &cudaSurface;
	}

	CudaSurface cudaSurface = CudaSurface(this->windowWidth, this->windowHeight);

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

	bool InitializeImGui(HWND hwnd);
	bool InitializeRayCastingTexture();
	bool InitializeRaytracingInteroperability();

	void raycastingRendering();


	void updateScene();

	// directx resources
	Microsoft::WRL::ComPtr<ID3D11Device>			device;// use to creat buffers
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>		deviceContext; //use to set resources for rendering
	Microsoft::WRL::ComPtr<IDXGISwapChain>			swapchain; // use to swap out our frame
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	renderTargetView; // where we are going to render our buffer
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	viewtofrontText; // where we are going to render our buffer


	// ImGui resoureces
	ImGuiContext* ImGuicontext = nullptr;

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



	// COM pointer to texture to store rendered bounding box
	Microsoft::WRL::ComPtr<ID3D11Texture2D> frontTex;	//front-face
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> frontTexResource;

	// Pointer to the adapter
	IDXGIAdapter* adapter;

	int windowWidth = 0;
	int windowHeight = 0;

	// Rendering Obejcts
	PathlineRenderer pathlineRenderer;
	StreamlineRenderer streamlineRenderer;
	BoxRenderer volumeBox;
	BoxRenderer seedBox;


	// Solver options
	bool streamline = true;
	bool pathline = false;



	Vertex* CudaVertex = nullptr;

	Interoperability cudaRayTracingInteroperability;

	Raycasting raycasting;

	RenderImGuiOptions renderImGuiOptions;

};