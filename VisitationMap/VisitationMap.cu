#include "VisitationMap.h"
#include "../Cuda/helper_math.h"
#include "../Raycaster/Raycasting_Helper.h"
#include "../VolumeIO/BinaryWriter.h"

bool VisitationMap::initialization()
{
	int3 gridSize = Array2Int3(solverOptions->gridSize);

	this->rays = (*this->width) * (*this->height);

	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;

	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;

	// initialize volume Input Output
	this->volume_IO.Initialize(this->fieldOptions);
	//this->updateFile_Single();
	a_visitationMap.initialize(gridSize.x, gridSize.y, gridSize.z);
	s_visitationMap.setInputArray(a_visitationMap.getArrayRef());
	s_visitationMap.initializeSurface();

	this->generateVisitationMap();

	updateEnsembleMember(); // Initialize texture_1 with a member 
	s_visitationMap.destroySurface();

	if (visitationOptions->saveVisitation)
	{
		this->copyToHost();
	}

	volumeTexture_0.setArray(a_visitationMap.getArrayRef());
	volumeTexture_0.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);


	return true;
}





bool VisitationMap::reinitialization()
{
	volumeTexture_1.release();
	volumeTexture_0.release();
	a_visitationMap.release();
	
	int3 gridSize = Array2Int3(solverOptions->gridSize);
	a_visitationMap.initialize(gridSize.x, gridSize.y, gridSize.z);

	s_visitationMap.setInputArray(a_visitationMap.getArrayRef());
	s_visitationMap.initializeSurface();


	this->generateVisitationMap();

	s_visitationMap.destroySurface();
	if (visitationOptions->saveVisitation)
	{
		this->copyToHost();
	}

	updateEnsembleMember(); // Initialize texture_1 with a member 
	volumeTexture_0.release();
	volumeTexture_0.setArray(a_visitationMap.getArrayRef());
	volumeTexture_0.initialize_array(false, cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder);

	return true;
}

bool VisitationMap::generateVisitationMap()
{
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(solverOptions->gridSize[2]); // Kernel calls are based on the Spanwise gridSize
	int3 gridSize = Array2Int3(solverOptions->gridSize);
	for (int memberIdx = fieldOptions[0].firstMemberIdx; memberIdx < fieldOptions[0].lastMemberIdx+1; memberIdx++)
	{
	
		loadTexture(gridSize, this->volumeTexture_0, volume_IO, solverOptions->currentIdx, fieldOptions->isCompressed);

		visitationMapGenerator << < blocks, thread >> > (*solverOptions, *raycastingOptions, *visitationOptions, volumeTexture_0.getTexture(), s_visitationMap.getSurfaceObject());
		this->volumeTexture_0.release();
	}

	
	return true;
}

bool VisitationMap::copyToHost()
{
	// Allocate 3D Array
	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
	cudaExtent extent = a_visitationMap.getExtent();
	float * h_field = (float*)malloc(sizeof(float4) *extent.depth*extent.height*extent.width);

	// set copy parameters to copy from velocity field to array
	cudaMemcpy3DParms cpyParams = { 0 };

	cpyParams.srcArray = a_visitationMap.getArrayRef();
	cpyParams.dstPtr = make_cudaPitchedPtr((void*)h_field, extent.width * sizeof(float4), extent.width, extent.height);
	cpyParams.kind = cudaMemcpyDeviceToHost;
	cpyParams.extent = extent;

	// Copy velocities to 3D Array
	gpuErrchk(cudaMemcpy3D(&cpyParams));
	// might need sync before release the host memory

	std::string outputFile = "v"+std::to_string(int(visitationOptions->visitationThreshold))+".bin";
	BinaryWriter binarywriter;
	//if (visitationOptions->visitation2D)
	//	outputFile += std::to_string(visitationOptions->visitationPlane) + ".bin";
	//else
	//	outputFile += ".bin";

	binarywriter.setFileName(outputFile.c_str());
	binarywriter.setFilePath((solverOptions->filePath +solverOptions->subpath + std::to_string(solverOptions->currentIdx)+"\\").c_str());
	binarywriter.setBuffer(reinterpret_cast<char*>(h_field));
	binarywriter.setBufferSize(extent.depth*extent.height*extent.width * sizeof(float4));
	binarywriter.write();


	free(h_field);

	return true;
}

void VisitationMap::updateEnsembleMember()
{

	int3 gridSize = Array2Int3(solverOptions->gridSize);
	this->volumeTexture_1.release();
	loadTexture(gridSize, this->volumeTexture_1,this->volume_IO,solverOptions->currentIdx,solverOptions->memberIdx,fieldOptions->isCompressed);

}


void VisitationMap::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());
	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(rays);
	int3 gridSize = Array2Int3(solverOptions->gridSize);


	switch (visitationOptions->raycastingMode)
	{
	case Visitation::RaycastingMode::DVR_VISITATION:

		CudaDVR_VISITATION << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture_0.getTexture(),
				this->volumeTexture_1.getTexture(),
				int(this->rays),
				*this->raycastingOptions,
				*this->renderingOptions,
				*this->visitationOptions
				);

		break;
	}

}

void VisitationMap::show(RenderImGuiOptions* renderImGuiOptions)
{
	if (renderImGuiOptions->showVisitationMap)
	{

		if (!this->visitationOptions->initialized)
		{

			this->initialization();
			this->visitationOptions->initialized = true;
		}

		if (renderImGuiOptions->updateRaycasting)
		{
			this->updateScene();
			renderImGuiOptions->updateRaycasting = false;

		}
		if (this->visitationOptions->resize)
		{
			//this->releaseRaycasting();
			//this->initializeRaycasting();
			//visitationOptions->resize = false;
		}

		// Overrided draw
		this->draw();

		if (visitationOptions->updateMember)
		{
			this->updateEnsembleMember();
			this->updateScene();
			visitationOptions->updateMember = false;
		}

		if ((renderImGuiOptions->updateVisitationMap && visitationOptions->applyUpdate) || visitationOptions->autoUpdate)
		{
			if (solverOptions->currentIdx < solverOptions->lastIdx)
			{
				solverOptions->currentIdx++;
			}
			else
			{
				visitationOptions->autoUpdate = false;
			}
			this->reinitialization();
			this->updateScene();
			renderImGuiOptions->updateVisitationMap = false;
			visitationOptions->applyUpdate = false;

		}
	}
}

