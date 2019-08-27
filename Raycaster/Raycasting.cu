#include "Raycasting.h"
#include "IsosurfaceHelperFunctions.h"
#include "Raycasting_Helper.h"


__constant__ BoundingBox d_boundingBox;


// Explicit instantiation
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Magnitude>	(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_X>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Y>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);
template __global__ void CudaIsoSurfacRenderer<struct IsosurfaceHelper::Velocity_Z>			(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance);


__host__ bool Raycasting::initialize()
{
	// set the number of rays = number of pixels
	this->rays = this->raycasting_desc.width * this->raycasting_desc.height;

	// initialize the bounding box
	initilizeBoundingBox();

	// Read and set field
	if(!fileLoaded)
	{
		this->volume_IO.Initialize(this->raycasting_desc.solverOption);
		this->initializeIO();
		this->initializeVolumeTexuture();

		fileLoaded = true;
	}
	if (fileChanged)
	{
		this->initializeIO();
		this->volumeTexture.release();
		this->initializeVolumeTexuture();

		fileChanged = false;
	}

	
	return true;

}

__host__ bool Raycasting::release()
{
	//cudaFree(d_boundingBox);
	//this->volumeTexture.release();

	return true;
}

__host__ void Raycasting::Rendering()
{


	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));


	CudaIsoSurfacRenderer<IsosurfaceHelper::Velocity_Magnitude> <<< blocks, thread >> >
		(
		this->raycastingSurface->getSurfaceObject(),
		this->volumeTexture.getTexture(),
		int(this->rays),
		20.0,
		.01f,
		1.0f
	);
}



__host__ bool Raycasting::initilizeBoundingBox()
{
	BoundingBox * h_boundingBox = new BoundingBox;

	h_boundingBox->eyePos = this->raycasting_desc.eyePos;
	h_boundingBox->viewDir = this->raycasting_desc.viewDir;
	h_boundingBox->upVec = this->raycasting_desc.upDir;

	h_boundingBox->width = this->raycasting_desc.width;
	h_boundingBox->height= this->raycasting_desc.height;
	h_boundingBox->gridDiameter = this->raycasting_desc.gridDiameter;
	h_boundingBox->updateBoxFaces();
	h_boundingBox->updateAspectRatio();
	h_boundingBox->constructEyeCoordinates();
	h_boundingBox->FOV = (this->raycasting_desc.FOV_deg) * 3.1415f / 180.0f;
	h_boundingBox->distImagePlane = 1;



	gpuErrchk(cudaMemcpyToSymbol(d_boundingBox, h_boundingBox, sizeof(BoundingBox)));
	

	delete h_boundingBox;
	
	return true;
}


__host__ bool Raycasting::initializeVolumeTexuture()
{
	this->volumeTexture.setSolverOptions(this->raycasting_desc.solverOption);
	this->volumeTexture.setField(this->raycasting_desc.field);
	this->volumeTexture.initialize();

	return true;
}

__host__ bool Raycasting::initializeIO()
{
	
	this->volume_IO.readVolume(this->raycasting_desc.solverOption->currentIdx);
	std::vector<char>* p_vec_buffer = volume_IO.flushBuffer();
	char* p_vec_buffer_temp = &(p_vec_buffer->at(0));
	raycasting_desc.field = reinterpret_cast<float*>(p_vec_buffer_temp);
	
	return true;
}







template <typename Observable>
__global__ void CudaIsoSurfacRenderer(cudaSurfaceObject_t raycastingSurface, cudaTextureObject_t field1, int rays, float isoValue, float samplingRate, float IsosurfaceTolerance)
{

	Observable observable;

	int index = blockIdx.x * blockDim.y * blockDim.x;
	index += threadIdx.y * blockDim.x;
	index += threadIdx.x;

	if (index < rays)
	{
		int2 pixel;
		pixel.y = index / d_boundingBox.width;
		pixel.x = index - pixel.y * d_boundingBox.width;
		float3 viewDir = d_boundingBox.viewDir;
		float3 pixelPos = pixelPosition(d_boundingBox, pixel.x, pixel.y);
		float2 NearFar = findIntersections(pixelPos, d_boundingBox);


		float3 rgb = { 1.0f, 0.3f, 0.0f };

		float3 positionOffset = d_boundingBox.gridDiameter / 2.0f;
		// if hits
		if (NearFar.y != -1)
		{
			float3 rayDir = normalize(pixelPos - d_boundingBox.eyePos);

			for (float t = NearFar.x; t < NearFar.y; t = t + samplingRate)
			{

				float3 relativePos = pixelPos + (rayDir * t);
				relativePos = (relativePos / d_boundingBox.gridDiameter) + make_float3(.5f, .5f, .5f);
				float4 velocity4D = tex3D<float4>(field1, relativePos.x, relativePos.y, relativePos.z);

				if (fabsf(observable.ValueAtXYZ(field1, relativePos) - isoValue) < IsosurfaceTolerance)
				{
					float3 gradient = observable.GradientAtXYZ(field1, relativePos, 0.01f);
					float diffuse = max(dot(normalize(gradient), viewDir), 0.0f);
					rgb = rgb * diffuse;
					float4 rgba = { rgb.x,rgb.y,rgb.z,0.0 };

					surf2Dwrite(rgbaFloatToUChar(rgba), raycastingSurface, 4 * pixel.x, pixel.y);
					break;
				}

			}



		}

	}


}
