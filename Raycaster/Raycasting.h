
#pragma once

#include "BoundingBox.h"

#include "..//VolumeTexture/VolumeTexture.h"
#include "..//Graphics/Camera.h"

#include "..//Graphics/ConstantBuffer.h"

#include "../Options/SolverOptions.h"
#include "../Options/RaycastingOptions.h"
#include "../Options/RenderingOptions.h"

#include<wrl/client.h>
#include <vector>

#include "..//Cuda/Interoperability.h"
#include "..//Cuda/CudaSurface.h"
#include "..//VolumeIO/Volume_IO_Z_Major.h"
#include "..//VolumeIO/Volume_IO_X_Major.h"

#include "cuda_runtime.h"
#include "..//Graphics/Shaders.h"
#include "..//Graphics/Vertex.h"
#include "..//Graphics/VertexBuffer.h"
#include "..//Options/DispresionOptions.h"
#include "..//Options/fluctuationheightfieldOptions.h"
#include "../Options/CrossSectionOptions.h"
#include "../Graphics/RenderImGuiOptions.h"




class Raycasting
{


protected:

	VolumeTexture3D volumeTexture;

	float* averageTemp = nullptr;
	float* d_averageTemp = nullptr;
	float FOV_deg	= 30.0f;
	float distImagePlane = 0.1f;
	unsigned int maxBlockDim = 16;

	int* width = nullptr;
	int* height = nullptr;

	size_t rays = 0;
	float3 gridCenter = { 0,0,0 };

	SolverOptions* solverOptions;
	RaycastingOptions* raycastingOptions;
	RenderingOptions* renderingOptions;
	
	float* field = nullptr;


	VertexShader vertexshader;
	PixelShader pixelshader;

	VertexBuffer<TexCoordVertex> vertexBuffer;


	Microsoft::WRL::ComPtr<ID3D11Texture2D>				raycastingTexture;
	Microsoft::WRL::ComPtr< ID3D11RenderTargetView>		renderTargetView;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState>		rasterizerstate;
	Microsoft::WRL::ComPtr<ID3D11SamplerState>			samplerState;	// For depth test between raycasting and line rendering
	Microsoft::WRL::ComPtr <ID3D11ShaderResourceView>	shaderResourceView;
	Microsoft::WRL::ComPtr<ID3D11BlendState>			blendState;

	ID3D11Device* device				= nullptr;
	IDXGIAdapter* pAdapter				= nullptr;
	ID3D11DeviceContext* deviceContext	= nullptr;
	Camera* camera						= nullptr;

	CudaSurface raycastingSurface;
	Interoperability interoperatibility;

	Volume_IO_Z_Major volume_IO;


	__host__ virtual bool initializeBoundingBox(); // Create and copy a Boundingbox in the Device constant memory
	__host__ virtual bool initializeShaders();

	// Interoperation methods (do not override or modify)
	__host__ bool initializeRaycastingTexture();
	__host__ bool initializeRaycastingInteroperability();
	__host__ bool createRaycastingShaderResourceView();
	__host__ bool initializeScene();
	__host__ bool initializeCudaSurface();
	__host__ bool initializeRasterizer();
	__host__ bool initializeSamplerstate();
	__host__ void setShaders();
	__host__ void updateFile
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);


	void loadTexture
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void loadTextureCompressed
	(
		SolverOptions * solverOptions,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);
	
public:

	__host__ virtual bool initialize
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);
	__host__ virtual bool release();
	__host__ virtual void rendering();
	__host__ virtual bool updateScene();


	__host__ bool resize();
	__host__  void draw();
	//__host__ void saveTexture();


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
		RenderingOptions* _renderingOptions,
		ID3D11Device* _device,
		IDXGIAdapter* _pAdapter,
		ID3D11DeviceContext* _deviceContext

	);

	__host__ virtual  void show(RenderImGuiOptions* renderImGuiOptions)
	{
		if (renderImGuiOptions->showRaycasting)
		{
			if (!this->raycastingOptions->initialized)
			{
				this->initialize(cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);
				this->raycastingOptions->initialized = true;
			}

			this->draw();

			if (renderImGuiOptions->updateRaycasting)
			{
				this->updateScene();

				renderImGuiOptions->updateRaycasting = false;

			}
			if (raycastingOptions->fileChanged)
			{
				this->updateFile();
				this->updateScene();
			}

			
		}
	}



};


template <typename Observable>
__global__ void CudaIsoSurfacRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	float isoValue,
	float samplingRate,
	float IsosurfaceTolerance
);

__global__ void CudaIsoSurfacRenderer_float
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	int3 gridSize,
	TimeSpace3DOptions timeSpace3DOptions
);

__global__ void CudaIsoSurfacRenderer_float_PlaneColor
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	int3 gridSize,
	TimeSpace3DOptions timeSpace3DOptions
);

__global__ void CudaIsoSurfacRenderer_float_PlaneColor
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	int3 gridSize,
	RaycastingOptions raycastingOptions
);

__global__ void CudaIsoSurfacRenderer_TurbulentDiffusivity
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	cudaTextureObject_t t_avg_temp,
	int rays,
	float isoValue,
	float samplingRate,
	float IsosurfaceTolerance,
	float * avg_temp
);



template <typename Observable>
__global__ void CudaIsoSurfacRendererSpaceTime
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	float isoValue,
	float samplingRate,
	float IsosurfaceTolerance
);





template <typename CrossSectionOptionsMode::SpanMode>
__global__ void CudaCrossSectionRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	cudaTextureObject_t gradient,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SolverOptions solverOptions,
	CrossSectionOptions crossSectionOptions
);



template <typename Observable>
__global__ void CudaTerrainRenderer
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
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
	DispersionOptions dispersionOptions,
	int traceTime
);


__global__ void CudaTerrainRenderer_Marching_extra
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
);


__global__ void CudaTerrainRenderer_Marching_extra_FSLE
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
);

__global__ void CudaTerrainRenderer_Marching_extra_FTLE_Color
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions
);


__global__ void CudaTerrainRenderer_extra_FTLE
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t extraField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
);



template <typename Observable>
__global__ void CudaTerrainRenderer_extra_double
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField_Primary,
	cudaTextureObject_t extraField_Primary,
	cudaTextureObject_t heightField_Secondary,
	cudaTextureObject_t extraField_Seconary,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	int traceTime
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


__global__ void CudaFilterExtremumX
(
	cudaSurfaceObject_t filtered,
	cudaTextureObject_t unfiltered,
	int2 Size,
	float threshold,
	int z
);



__device__ float3 binarySearch_tex1D
(
	cudaTextureObject_t field,
	float3& _position,
	float3& gridDiameter,
	int3& gridSize,
	float3& _samplingStep,
	float& value,
	float& tolerance,
	int maxIteration
);