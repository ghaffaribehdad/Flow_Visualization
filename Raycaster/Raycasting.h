
#pragma once

#include "BoundingBox.h"

#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Graphics/Camera.h"

#include "..//Graphics/ConstantBuffer.h"
#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include<wrl/client.h>
#include <DirectXMath.h>
#include <vector>

#include "..//Cuda/Interoperability.h"
#include "..//Cuda/CudaSurface.h"
#include "..//Volume/Volume_IO.h"

#include "cuda_runtime.h"
#include "..//Graphics/Shaders.h"
#include "..//Graphics/Vertex.h"
#include "..//Graphics/VertexBuffer.h"



inline float3 XMFloat3ToFloat3(const DirectX::XMFLOAT3& src)
{
	return make_float3(src.x, src.y, src.z);
}

inline float3 ArrayFloat3ToFloat3(float* src)
{
	return make_float3(src[0], src[1], src[2]);
}

inline int3 ArrayInt3ToInt3(int* src)
{
	return make_int3(src[0], src[1], src[2]);
}




class Raycasting
{

private:


	float FOV_deg	= 30.0f;
	unsigned int maxBlockDim = 32;
	int* width = nullptr;
	int* height = nullptr;

	size_t rays = 0;
	float3 gridCenter = { 0,0,0 };

	SolverOptions* solverOptions;
	RaycastingOptions* raycastingOptions;
	
	BoundingBox* d_BoundingBox;
	float* field = nullptr;


	VertexShader vertexshader;
	PixelShader pixelshader;

	VertexBuffer<TexCoordVertex> vertexBuffer;


	Microsoft::WRL::ComPtr<ID3D11Texture2D>			raycastingTexture;
	Microsoft::WRL::ComPtr< ID3D11RenderTargetView> renderTargetView;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerstate;
	Microsoft::WRL::ComPtr<ID3D11SamplerState>		samplerState;	// For depth test between raycasting and line rendering
	Microsoft::WRL::ComPtr <ID3D11ShaderResourceView> shaderResourceView;

	ID3D11Device* device		= nullptr;
	IDXGIAdapter* pAdapter		= nullptr;
	ID3D11DeviceContext* deviceContext	= nullptr;


	VolumeTexture volumeTexture;
	Volume_IO volume_IO;
	CudaSurface raycastingSurface;
	Interoperability interoperatibility;
	Camera* camera;

	// in case of resizing

	__host__ bool initializeBoundingBox();
	__host__ bool initializeIO();
	__host__ bool initializeVolumeTexuture();
	__host__ bool initializeRaycastingTexture();
	__host__ bool initializeRaycastingInteroperability();
	__host__ bool initializeCudaSurface();
	__host__ bool initializeRasterizer();
	__host__ bool initializeSamplerstate();
	__host__ bool createRaycastingShaderResourceView();
	__host__ bool initializeScene();
	__host__ bool initializeShaders();
	__host__ void setShaders();

public:

	__host__ bool initialize();
	__host__ bool release();
	__host__ void rendering();
	//__host__ void saveTexture();
	__host__ bool updateScene();
	__host__ bool resize();
	

	__host__ void draw();



	__host__ ID3D11Texture2D * getTexture()
	{
		return this->raycastingTexture.Get();
	}

	__host__ void setResources
	(
		Camera* _camera,
		int * _width,
		int * _height,
		SolverOptions* _solverOption,
		RaycastingOptions* _raycastingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext

	);



};


template <typename Observable>
__global__ void CudaIsoSurfacRenderer(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
