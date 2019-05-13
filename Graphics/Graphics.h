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
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"



class Graphics
{
public:
	bool Initialize(HWND hwnd, int width, int height);

	// main rendering function
	void RenderFrame();

	// Add Camera object
	Camera camera;
private:

	// call by Initialize() funcion
	bool InitializeDirectX(HWND hwnd);
	bool InitializeShaders();
	bool InitializeScene();

	// directx resources
	Microsoft::WRL::ComPtr<ID3D11Device>			device;// use to creat buffers
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>		deviceContext; //use to set resources for rendering
	Microsoft::WRL::ComPtr<IDXGISwapChain>			swapchain; // use to swap out our frame
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView>	renderTargetView; // where we are going to render our buffer

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
	Microsoft::WRL::ComPtr<ID3D11SamplerState> samplerState;

	// COM pointer to texture
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> myTexture;

	int windowWidth = 0;
	int windowHeight = 0;

	Timer fpsTimer;


};