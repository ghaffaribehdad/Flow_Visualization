
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
#include "..//Options/SpaceTimeOptions.h"
#include "../Options/CrossSectionOptions.h"
#include "../Graphics/RenderImGuiOptions.h"
#include "../Cuda/CudaArray.h"





class Raycasting
{


protected:

	VolumeTexture3D volumeTexture_0;
	VolumeTexture3D volumeTexture_1;


	VolumeTexture3D volumeTexture_L1;
	VolumeTexture3D volumeTexture_L2;
	

	float* averageTemp = nullptr;
	float* d_averageTemp = nullptr;
	double FOV_deg = 40.0;
	float distImagePlane = 0.01f;
	unsigned int maxBlockDim = 16;

	bool b_initialized = false;
	bool b_updateScene = false;

	int* width = nullptr;
	int* height = nullptr;

	size_t rays = 0;
	float3 gridCenter = { 0,0,0 };

	SolverOptions* solverOptions;
	RaycastingOptions* raycastingOptions;
	RenderingOptions* renderingOptions;

	FieldOptions * fieldOptions;

	float* field = nullptr;
	CudaArray_3D<float4> a_mipmap_L1;
	CudaArray_3D<float4> a_mipmap_L2;

	VertexShader vertexshader;
	PixelShader pixelshader;

	VertexBuffer<TexCoordVertex> vertexBuffer;

	ConstantBuffer<CB_pixelShader_Sampler>				PS_constantBuffer;
	Microsoft::WRL::ComPtr<ID3D11Texture2D>				raycastingTexture;
	Microsoft::WRL::ComPtr< ID3D11RenderTargetView>		renderTargetView;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState>		rasterizerstate;
	Microsoft::WRL::ComPtr<ID3D11SamplerState>			samplerState;	// For depth test between raycasting and line rendering
	Microsoft::WRL::ComPtr <ID3D11ShaderResourceView>	shaderResourceView;
	Microsoft::WRL::ComPtr<ID3D11BlendState>			blendState;

	ID3D11Device* device = nullptr;
	IDXGIAdapter* pAdapter = nullptr;
	ID3D11DeviceContext* deviceContext = nullptr;
	Camera* camera = nullptr;

	CudaSurface raycastingSurface;
	CudaSurface s_mipmapped;
	Interoperability interoperatibility;

	Volume_IO_Z_Major volume_IO_Primary;
	Volume_IO_Z_Major volume_IO_Secondary;


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
	__host__ void updateFile_Single
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);

	__host__ void updateFile_Double
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);

	__host__ void updateFile_MultiScale
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


	void loadTexture
	(
		int3 & gridSize,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void loadTexture
	(
		int3 & gridSize,
		VolumeTexture3D & volumeTexture,
		Volume_IO_Z_Major & volumeIO,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void loadTextureCompressed
	(
		int3 & gridSize,
		VolumeTexture3D & volumeTexture,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void loadTextureCompressed
	(
		int3 & gridSize,
		VolumeTexture3D & volumeTexture,
		Volume_IO_Z_Major & volumeIO,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void loadTextureCompressed_double
	(
		int * gridSize_0,
		int * gridSize_1,
		VolumeTexture3D & volumeTexture_0,
		VolumeTexture3D & volumeTexture_1,
		const int & idx,
		cudaTextureAddressMode addressModeX,
		cudaTextureAddressMode addressModeY,
		cudaTextureAddressMode addressModeZ
	);

	void generateMipmapL1();
	void initializeMipmapL1();
	void initializeMipmapL2();
	void generateMipmapL2();

public:


	

	__host__ virtual bool initialize_Single
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);

	__host__ virtual bool initialize_Double
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);

	__host__ virtual bool initialize_Multiscale
	(
		cudaTextureAddressMode addressMode_X = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Y = cudaAddressModeBorder,
		cudaTextureAddressMode addressMode_Z = cudaAddressModeBorder
	);

	__host__ virtual bool release();
	__host__ virtual void rendering();
	__host__ virtual bool updateScene();
	__host__ virtual bool updateconstantBuffer()
	{
		PS_constantBuffer.data.transparency = raycastingOptions->transparency_0;
		PS_constantBuffer.ApplyChanges();

		return true;
	}

	__host__ bool initializeBuffers()
	{
		HRESULT	hr = this->PS_constantBuffer.Initialize(this->device, this->deviceContext);
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create Pixel shader Constant buffer.");
			return false;
		}
		return true;
	}

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
		ID3D11DeviceContext* _deviceContext,
		FieldOptions * fieldOptions

	);



	__host__ bool resize()
	{
		this->raycastingTexture->Release();
		this->initializeRaycastingTexture();

		this->raycastingSurface.destroySurface();
		this->interoperatibility.release();

		this->initializeRaycastingInteroperability();
		this->initializeCudaSurface();
		this->initializeBoundingBox();
		this->rendering();
		this->interoperatibility.release();

		return true;
	}

	__host__ virtual void show(RenderImGuiOptions* renderImGuiOptions)
	{
		if (b_initialized && !renderImGuiOptions->showRaycasting)
		{
			this->volumeTexture_0.release();
			this->volumeTexture_1.release();
			this->raycastingTexture.Reset();
			this->volume_IO_Primary.release();
			this->volume_IO_Secondary.release();
			this->b_initialized = false;
		}
		if (renderImGuiOptions->showRaycasting)
		{

			switch (raycastingOptions->raycastingMode)
			{
			case RaycastingMode::Mode::SINGLE:

				if (!b_initialized)
				{
					this->initialize_Single();
				}

				if (raycastingOptions->fileChanged || renderImGuiOptions->fileChanged)
				{
					this->updateFile_Single();
					renderImGuiOptions->fileChanged = false;
				}
				if (renderImGuiOptions->updateRaycasting)
				{
					this->updateScene();
					renderImGuiOptions->updateRaycasting = false;

				}

				this->draw();

				break;

			case RaycastingMode::Mode::DOUBLE:
			case RaycastingMode::Mode::DOUBLE_SEPARATE:
			case RaycastingMode::Mode::DOUBLE_ADVANCED:
			case RaycastingMode::Mode::DOUBLE_TRANSPARENCY:

				if (!b_initialized)
				{
					this->initialize_Double();
				}

				if (raycastingOptions->fileChanged || renderImGuiOptions->fileChanged)
				{
					this->updateFile_Double();
					renderImGuiOptions->fileChanged = false;
				}
				if (renderImGuiOptions->updateRaycasting)
				{
					this->updateScene();
					renderImGuiOptions->updateRaycasting = false;

				}

				this->draw();

				break;

			case  RaycastingMode::Mode::MULTISCALE:
			case RaycastingMode::Mode::MULTISCALE_TEMP:

				if (!b_initialized)
				{
					this->initialize_Multiscale();
				}

				if (raycastingOptions->fileChanged || renderImGuiOptions->fileChanged)
				{
					this->updateFile_MultiScale();
					renderImGuiOptions->fileChanged = false;
				}
				if (renderImGuiOptions->updateRaycasting)
				{
					this->updateScene();
					renderImGuiOptions->updateRaycasting = false;

				}

				this->draw();

				break;


			case  RaycastingMode::Mode::MULTISCALE_DEFECT:

				if (!b_initialized)
				{
					this->initialize_Multiscale();
				}

				if (raycastingOptions->fileChanged || renderImGuiOptions->fileChanged)
				{
					this->updateFile_MultiScale();
					renderImGuiOptions->fileChanged = false;
				}
				if (renderImGuiOptions->updateRaycasting)
				{
					this->updateScene();
					renderImGuiOptions->updateRaycasting = false;

				}

				this->draw();

				break;


			case  RaycastingMode::Mode::PLANAR:
			case RaycastingMode::Mode::PROJECTION_BACKWARD:   
			case RaycastingMode::Mode::PROJECTION_FORWARD:   
			case RaycastingMode::Mode::PROJECTION_AVERAGE:   
			case RaycastingMode::Mode::PROJECTION_LENGTH:   

				if (!b_initialized)
				{
					this->initialize_Single();
				}

				if (raycastingOptions->fileChanged || renderImGuiOptions->fileChanged)
				{
					this->updateFile_Single();
					renderImGuiOptions->fileChanged = false;
				}
				if (renderImGuiOptions->updateRaycasting)
				{
					this->updateScene();
					renderImGuiOptions->updateRaycasting = false;

				}

				this->draw();

				break;


			}
		}
	}
};




template <typename Observable>
__global__ void CudaIsoSurfacRendererAnalytic
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t field1,
	int rays,
	RaycastingOptions raycastingOptions
);




//template <typename Observable>
//__global__ void CudaIsoSurfacRendererSpaceTime
//(
//	cudaSurfaceObject_t raycastingSurface,
//	cudaTextureObject_t field1,
//	int rays,
//	float isoValue,
//	float samplingRate,
//	float IsosurfaceTolerance
//);





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
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	DispersionOptions dispersionOptions,
	RenderingOptions renderingOptions,
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






template <typename Observable1, typename Observable2>
__global__ void CudaTerrainRenderer_extra_fluctuation
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions fluctuationOptions,
	RenderingOptions renderingOptions
);




__global__ void CudaTerrainRenderer_height_isoProjection
(
	cudaSurfaceObject_t raycastingSurface,
	cudaTextureObject_t heightField,
	cudaTextureObject_t field,
	int rays,
	float samplingRate,
	float IsosurfaceTolerance,
	SpaceTimeOptions fluctuationOptions,
	RenderingOptions renderingOptions
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


