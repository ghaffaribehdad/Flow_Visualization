#include "pathSpaceTime.h"
#include "DispersionHelper.h"
#include "..//Cuda/helper_math.h"
#include "..//ErrorLogger/ErrorLogger.h"
#include "..//Cuda/Cuda_helper_math_host.h"
#include "..//Raycaster/IsosurfaceHelperFunctions.h"
#include "..//Cuda/CudaHelperFunctions.h"
#include "..//Raycaster/Raycasting_Helper.h"
#include "..//Cuda/CudaSolver.h"
#include "../Particle/ParticleHelperFunctions.h"
extern __constant__  BoundingBox d_boundingBox;
extern __constant__  BoundingBox d_boundingBox_spacetime;




bool PathSpaceTime::initialize(
	cudaTextureAddressMode addressMode_X,
	cudaTextureAddressMode addressMode_Y,
	cudaTextureAddressMode addressMode_Z){


	int time_interval = pathSpaceTimeOptions->lastIdx - pathSpaceTimeOptions->firstIdx;
	this->m_gridSize3D =
	{
		this->pathSpaceTimeOptions->seedGrid[0],
		this->pathSpaceTimeOptions->seedGrid[1],
		this->pathSpaceTimeOptions->seedGrid[2] * pathSpaceTimeOptions->timeGrid
	};

	
	//m_gridSize3D.x = m_gridSize3D.x * (pathSurfaceOptions->lastIdx - pathSurfaceOptions->firstIdx);
	//pathSurfaceOptions->minimumDim = minDim(m_gridSize3D);
	//switch (pathSurfaceOptions->minimumDim) {
	//case 0:
	//	m_gridSize3D.x = m_gridSize3D.x * timeDim;
	//	break;
	//case 1:
	//	m_gridSize3D.y = m_gridSize3D.y * timeDim;
	//	break;
	//case 2:
	//	m_gridSize3D.z = m_gridSize3D.z * timeDim;
	//	break;
	//}
	

	if (!this->initializeRaycastingTexture())					// initialize texture (the texture we need to write to)
		return false;


	if (!this->initializeBoundingBox())							// initialize the bounding box ( copy data to the constant memory of GPU about Bounding Box)
		return false;


	this->rays = (*this->width) * (*this->height);				// Set number of rays based on the number of pixels

	// initialize volume Input Output
	this->volume_IO.Initialize(&this->fieldOptions[0]);


	// Initialize Height Field as an empty CUDA array 3D
	if (!a_PathSpaceTime.initialize(m_gridSize3D))
		return false;


	cudaArray_t pCudaArray = a_PathSpaceTime.getArray();
	s_PathSpaceTime.setInputArray(pCudaArray);
	if (!this->s_PathSpaceTime.initializeSurface())
		return false;

	// Initialize positions at t = 0
	if (!this->InitializePathSurface())
		return false;

	// Trace the fluctuation field
	this->generatePathField3D();

	// Destroy the surface
	this->s_PathSpaceTime.destroySurface();

	volumeTexture3D_height.setArray(a_PathSpaceTime.getArrayRef());

	this->volumeTexture3D_height.initialize_array(false, cudaAddressModeWrap, cudaAddressModeWrap, cudaAddressModeWrap);

	return true;
}

bool PathSpaceTime::InitializePathSurface() {
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(this->pathSpaceTimeOptions->seedGrid[0] * this->pathSpaceTimeOptions->seedGrid[1] * this->pathSpaceTimeOptions->seedGrid[2]); // Kernel calls are based on the Spanwise gridSize
	int n_particles = pathSpaceTimeOptions->seedGrid[0] * pathSpaceTimeOptions->seedGrid[1] * pathSpaceTimeOptions->seedGrid[2];
	size_t Particles_byte = sizeof(Particle) * n_particles;
	Particle* hh_particle = new Particle[n_particles];
	Particle* dd_particle = nullptr;
	seedParticleGridPoints(hh_particle, solverOptions->gridDiameter, solverOptions->gridDiameter, solverOptions->seedBoxPos, pathSpaceTimeOptions->seedGrid);
	gpuErrchk(cudaMalloc((void**)&dd_particle, Particles_byte));
	gpuErrchk(cudaMemcpy((void*)dd_particle, (void*)hh_particle, Particles_byte, cudaMemcpyHostToDevice));
	//for (int i = 0; i < n_particles; i++)
	//{
	//	printf("particle %d is at (%3f,%3f,%3f)\n", i, hh_particle[i].m_position.x, hh_particle[i].m_position.y, hh_particle[i].m_position.z);
	//}
	initializePathSpaceTime << < blocks, thread >> > (dd_particle, s_PathSpaceTime.getSurfaceObject(), *pathSpaceTimeOptions);
	
	// Release particle for host and device
	delete[] hh_particle;
	cudaFree(dd_particle);

	return true;
}
void PathSpaceTime::generatePathField3D()
{
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = BLOCK_THREAD(this->pathSpaceTimeOptions->seedGrid[0] * this->pathSpaceTimeOptions->seedGrid[1] * this->pathSpaceTimeOptions->seedGrid[2]); // Kernel calls are based on the Spanwise gridSize
	Timer timer;
	int tStep = 0;

	for (int t = pathSpaceTimeOptions->firstIdx; t < pathSpaceTimeOptions->lastIdx; t++) {

		switch (fieldOptions->isCompressed)
		{

		case true: // Compressed Data
		
			if (tStep == 0)
			{
				this->volume_IO.readVolume_Compressed(t, Array2Int3(fieldOptions->gridSize));
				float * d_VelocityField = this->volume_IO.getField_float_GPU();
				t_velocityField_0.setField(d_VelocityField);
				TIMELAPSE(t_velocityField_0.initialize_devicePointer(Array2Int3(fieldOptions->gridSize)), "Initialize Texture including DDCopy");
				cudaFree(d_VelocityField);

				this->volume_IO.readVolume_Compressed(t+1, Array2Int3(fieldOptions->gridSize));
				d_VelocityField = this->volume_IO.getField_float_GPU();
				t_velocityField_1.setField(d_VelocityField);
				TIMELAPSE(t_velocityField_1.initialize_devicePointer(Array2Int3(fieldOptions->gridSize)), "Initialize Texture including DDCopy");
				cudaFree(d_VelocityField);
			}
			else if (tStep % 2 == 0) // => EVEN
			{
				t_velocityField_1.release();
				this->volume_IO.readVolume_Compressed(t + 1, Array2Int3(fieldOptions->gridSize));
				float * d_VelocityField = this->volume_IO.getField_float_GPU();
				t_velocityField_1.setField(d_VelocityField);
				TIMELAPSE(t_velocityField_1.initialize_devicePointer(Array2Int3(fieldOptions->gridSize)), "Initialize Texture including DDCopy");
				cudaFree(d_VelocityField);

			}
			else if (tStep % 2 != 0) // => ODD
			{
				t_velocityField_0.release();
				this->volume_IO.readVolume_Compressed(t+1, Array2Int3(fieldOptions->gridSize));
				float * d_VelocityField = this->volume_IO.getField_float_GPU();
				t_velocityField_0.setField(d_VelocityField);
				TIMELAPSE(t_velocityField_0.initialize_devicePointer(Array2Int3(fieldOptions->gridSize)), "Initialize Texture including DDCopy");
				cudaFree(d_VelocityField);

			}
			break;
		

		case false: // Uncompressed Data
			assert(false);
			break;

		}

		printf("Time step is %d",tStep);

		TracingPathSurface << < blocks, thread >> > (
			t_velocityField_0.getTexture(),
			t_velocityField_1.getTexture(),
			s_PathSpaceTime.getSurfaceObject(),
			*this->solverOptions,
			*this->pathSpaceTimeOptions,
			tStep);
		tStep++;
	}

	volume_IO.release();
	solverOptions->compressResourceInitialized = false;
}



__host__ void PathSpaceTime::rendering()
{
	this->deviceContext->PSSetSamplers(0, 1, this->samplerState.GetAddressOf());




	this->deviceContext->ClearRenderTargetView(this->renderTargetView.Get(), renderingOptions->bgColor);// Clear the target view

	// Calculates the block and grid sizes
	unsigned int blocks;
	dim3 thread = { maxBlockDim,maxBlockDim,1 };
	blocks = static_cast<unsigned int>((this->rays % (thread.x * thread.y) == 0 ? rays / (thread.x * thread.y) : rays / (thread.x * thread.y) + 1));

	if (pathSpaceTimeOptions->colorCoding)
	{
		CudaIsoSurfacRenderer_Planar_PathSpace << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture3D_height.getTexture(),
				int(this->rays),
				*pathSpaceTimeOptions,
				*raycastingOptions,
				*renderingOptions
				);
	}
	else {
		CudaIsoSurfacRenderer_Single << < blocks, thread >> >
			(
				this->raycastingSurface.getSurfaceObject(),
				this->volumeTexture3D_height.getTexture(),
				int(this->rays),
				*raycastingOptions,
				*renderingOptions,
				*pathSpaceTimeOptions
				);
	}


}