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
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "RenderImGui.h"
#include "..\\Cuda\Interoperability.cuh"
#include <d3d11.h>
#include "..\\Cuda\cudaSurface.cuh"
#include "..\\testCudaInterOp.cuh"



class Graphics
{
	friend CUDASolver<float>;
	friend CUDASolver<double>;

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

	// Getter Functions
	IDXGIAdapter* GetAdapter();
	ID3D11Device* GetDevice();
	ID3D11Buffer* GetVertexBuffer();

	// Setter Functions
	void setCudaVertex(void* cudaVertex, size_t size)
	{
		cudaMemcpy(vertexBuffer.Get(), cudaVertex, size, cudaMemcpyDeviceToDevice);
	}

	bool showLines = false;

	bool initializeRaycasting();
	bool releaseRaycastingResource()
	{
		// destroy and release the resources
		//cudaSurface.destroySurface();
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

	CudaSurface * getSurfaceObject()
	{
		return &cudaSurface;
	}

private:

	// camera propertis
	float FOV = 90.0;


	// call by Initialize() funcion
	bool InitializeDirectX(HWND hwnd);
	bool InitializeResources();
	bool InitializeShaders();
	bool InitializeScene();
	bool InitializeImGui(HWND hwnd);
	bool InitializeRayCastingTexture();
	bool InitializeRaytracingInteroperability();


	// directx resources
	Microsoft::WRL::ComPtr<ID3D11Device>			device;// use to creat buffers
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>		deviceContext; //use to set resources for rendering
	Microsoft::WRL::ComPtr<IDXGISwapChain>			swapchain; // use to swap out our frame
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	renderTargetView; // where we are going to render our buffer
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	viewtofrontText; // where we are going to render our buffer

	
	// ImGui resoureces
	ImGuiContext * ImGuicontext = nullptr;
	
	// Shaders
	VertexShader vertexshader;
	PixelShader pixelshader;

	// Buffers
	VertexBuffer<Vertex> vertexBuffer;
	IndexBuffer indexBuffer;
	ConstantBuffer<CB_VS_vertexshader> constantBuffer;

	// Depth stencil view and buffer and state
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView>		depthStencilView;
	Microsoft::WRL::ComPtr<ID3D11Texture2D>				depthStencilBuffer;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState>		depthStencilState;


	// Resterizer com pointer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;

	// Standart unique pointer to sprite batch/font
	std::unique_ptr<DirectX::SpriteBatch>	spriteBatch;
	std::unique_ptr<DirectX::SpriteFont>	spriteFont;

	// COM pointer to sampler state
	//Microsoft::WRL::ComPtr<ID3D11SamplerState> samplerState;

	// COM pointer to texture to store rendered bounding box
	Microsoft::WRL::ComPtr<ID3D11Texture2D> frontTex;	//front-face
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> frontTexResource;

	// Pointer to the adapter
	IDXGIAdapter* adapter;

	int windowWidth = 0;
	int windowHeight = 0;

	Timer fpsTimer;


	void RenderImGui();

	// Solver options
	bool streamline = true;
	bool pathline = false;

	// Log pointer
	char* log = new char[100];

	Vertex* CudaVertex = nullptr;

	Interoperability cudaRayTracingInteroperability;
	CudaSurface cudaSurface = CudaSurface(this->windowWidth, this->windowHeight);



};