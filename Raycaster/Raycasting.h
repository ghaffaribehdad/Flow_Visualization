
#pragma once

#include "BoundingBox.h"

#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Graphics/Camera.h"

#include "..//Graphics/ConstantBuffer.h"
#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include<wrl/client.h>
#include <vector>

#include "..//Cuda/Interoperability.h"
#include "..//Cuda/CudaSurface.h"
#include "..//Volume/Volume_IO.h"

#include "cuda_runtime.h"
#include "..//Graphics/Shaders.h"
#include "..//Graphics/Vertex.h"
#include "..//Graphics/VertexBuffer.h"
#include "..//Options/DispresionOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"




class Raycasting
{


private:
	VolumeTexture3D volumeTexture;

protected:



	float FOV_deg	= 30.0f;
	float distImagePlane = 0.1f;
	unsigned int maxBlockDim = 16;

	int* width = nullptr;
	int* height = nullptr;

	size_t rays = 0;
	float3 gridCenter = { 0,0,0 };

	SolverOptions* solverOptions;
	RaycastingOptions* raycastingOptions;
	
	float* field = nullptr;


	VertexShader vertexshader;
	PixelShader pixelshader;

	VertexBuffer<TexCoordVertex> vertexBuffer;


	Microsoft::WRL::ComPtr<ID3D11Texture2D>			raycastingTexture;
	Microsoft::WRL::ComPtr< ID3D11RenderTargetView> renderTargetView;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState>	rasterizerstate;
	Microsoft::WRL::ComPtr<ID3D11SamplerState>		samplerState;	// For depth test between raycasting and line rendering
	Microsoft::WRL::ComPtr <ID3D11ShaderResourceView> shaderResourceView;

	ID3D11Device* device		= nullptr;
	IDXGIAdapter* pAdapter		= nullptr;
	ID3D11DeviceContext* deviceContext	= nullptr;
	Camera* camera;
	CudaSurface raycastingSurface;
	Interoperability interoperatibility;
	volumeIO::Volume_IO volume_IO;




	__host__ virtual bool initializeBoundingBox(); // Create and copy a Boundingbox in the Device constant memory
	__host__ bool initializeIO();
	__host__ bool initializeVolumeTexuture(cudaTextureAddressMode , cudaTextureAddressMode, cudaTextureAddressMode);
	__host__ bool initializeVolumeTexuture(cudaTextureAddressMode , cudaTextureAddressMode, cudaTextureAddressMode, VolumeTexture3D & volumeTexture);
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

	__host__ virtual bool initialize(cudaTextureAddressMode, cudaTextureAddressMode, cudaTextureAddressMode);
	__host__ virtual bool release();
	__host__ virtual void rendering();
	//__host__ void saveTexture();
	__host__ virtual bool updateScene();
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

template <typename Observable>
__global__ void CudaTerrainRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
);


template <typename Observable>
__global__ void CudaTerrainRenderer_extra
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
);


template <typename Observable>
__global__ void CudaTerrainRenderer_extra_fluctuation
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	FluctuationheightfieldOptions fluctuationOptions
);