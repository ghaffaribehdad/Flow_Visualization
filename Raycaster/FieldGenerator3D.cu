#include "FieldGenerator3D.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include <cuda_runtime.h>
#include "..//Raycaster/Raycasting_Helper.h"


//explicit instantiation

bool FieldGenerator3D::retrace()
{



	// Initialize Height Field as an empty cuda array 3D
	if (!this->generateVolumetricField())
		return false;

	return true;
}

bool FieldGenerator3D::initialize
(
	cudaTextureAddressMode addressMode_X ,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z 
)
{

	if (!this->initializeRaycastingTexture())				// initilize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())		// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	// set the number of rays = number of pixels
	this->rays = (*this->width) * (*this->height);

	volume_IO.Initialize(this->solverOptions);

	generateVolumetricField();

	return true;
}





__host__ bool FieldGenerator3D::InitializeSurface3D()
{
	// Assign the hightArray to the hightSurface and initialize the surface
	this->cudaSurface.setInputArray(this->cudaArray3D.getArrayRef());
	if (!this->cudaSurface.initializeSurface())
		return false;

	return true;
}




// Release resources 
bool FieldGenerator3D::release()
{
	Raycasting::release();
	cudaDestroyTextureObject(this->cudaTexture3D_float.getTexture());
	this->cudaArray3D.release();

	return true;
}






__host__ void FieldGenerator3D::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	//this->deviceContext->OMSetBlendState(this->blendState.Get(), NULL, 0xFFFFFFFF);

	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view


	cudaRaycaster();



};


bool FieldGenerator3D::updateScene()
{
	if (!this->initializeRaycastingInteroperability())	// Create interoperability while we need to release it at the end of rendering
		return false;

	if (!this->initializeCudaSurface())					// reinitilize cudaSurface	
		return false;

	if (!this->initializeBoundingBox())					//updates constant memory
		return false;


	this->rendering();


	if (!this->raycastingSurface.destroySurface())
		return false;

	this->interoperatibility.release();


	return true;
}

