#include "Pathlinesolver.cuh"

// Explicit instantions
template class PathlineSolver<float>;
//template class PathlineSolver<double>;


//
//template <typename T>
//void PathlineSolver<T>::InitializeParticles()
//{
//	// Create an array of particles
//	this->h_Particles = new Particle<T>[solverOptions.lines_count];
//
//	float3 gridDiameter =
//	{
//		solverOptions.gridDiameter[0],
//		solverOptions.gridDiameter[1],
//		solverOptions.gridDiameter[2]
//	};
//	// Seed Particles Randomly according to the grid diameters
//	for (int i = 0; i < solverOptions.lines_count; i++)
//	{
//		h_Particles[i].seedParticle(gridDiameter);
//	}
//
//	size_t Particles_byte = sizeof(*h_Particles) * solverOptions.lines_count;
//
//	// Upload Velocity Filled to GPU 
//
//	gpuErrchk(cudaMalloc((void**)& d_Particles, Particles_byte));
//
//	gpuErrchk(cudaMemcpy(d_Particles, h_Particles, Particles_byte, cudaMemcpyHostToDevice));
//
//	delete h_Particles;
//}
//
//
//template <typename T>
//__host__ bool PathlineSolver<T>::InitializeTexture()
//{
//
//	// Cuda 3D array of velocities
//	cudaArray_t cuArray_velocity;
//
//
//	// define the size of the velocity field
//	cudaExtent extent =
//	{
//		static_cast<size_t>(solverOptions.gridSize[0]),
//		static_cast<size_t>(solverOptions.gridSize[1]),
//		static_cast<size_t>(solverOptions.gridSize[2])
//	};
//
//
//	// Allocate 3D Array
//	cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
//	gpuErrchk(cudaMalloc3DArray(&cuArray_velocity, &channelFormatDesc, extent, 0));
//
//
//
//	// set copy parameters to copy from velocity field to array
//	cudaMemcpy3DParms cpyParams = { 0 };
//
//	cpyParams.srcPtr = make_cudaPitchedPtr((void*)this->h_VelocityField, extent.width * sizeof(float4), extent.height, extent.depth);
//	cpyParams.dstArray = cuArray_velocity;
//	cpyParams.kind = cudaMemcpyHostToDevice;
//	cpyParams.extent = extent;
//
//
//	// Copy velocities to 3D Array
//	gpuErrchk(cudaMemcpy3D(&cpyParams));
//	// might need sync before release the host memory
//
//	// Release the Volume while it is copied on GPU
//	this->volume_IO.release();
//
//
//	// Set Texture Description
//	cudaTextureDesc texDesc;
//	cudaResourceDesc resDesc;
//	cudaResourceViewDesc resViewDesc;
//
//	memset(&resDesc, 0, sizeof(resDesc));
//	memset(&texDesc, 0, sizeof(texDesc));
//	memset(&resViewDesc, 0, sizeof(resViewDesc));
//
//
//
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = cuArray_velocity;
//
//	// Texture Description
//	texDesc.normalizedCoords = true;
//	texDesc.filterMode = cudaFilterModeLinear;
//	texDesc.addressMode[0] = cudaAddressModeClamp;
//	texDesc.addressMode[1] = cudaAddressModeClamp;
//	texDesc.addressMode[2] = cudaAddressModeClamp;
//	texDesc.readMode = cudaReadModeElementType;
//
//
//
//	// Create the texture and bind it to the array
//	gpuErrchk(cudaCreateTextureObject(&this->t_VelocityField, &resDesc, &texDesc, NULL));
//
//	cudaDestroyTextureObject(t_VelocityField);
//	return true;
//
//}
//
//template <typename T>
//__global__ void TracingParticles(Particle<T>* d_particles, cudaTextureObject_t t_VelocityField, SolverOptions solverOptions, Vertex* p_VertexBuffer)
//{
//	int index = blockDim.x * blockIdx.x + threadIdx.x;
//	if (index < solverOptions.lineLength)
//	{
//		int lineLength = solverOptions.lineLength;
//		int index_buffer = index * lineLength;
//		float dt = solverOptions.dt;
//		float3 gridDiameter =
//		{
//			solverOptions.gridDiameter[0],
//			solverOptions.gridDiameter[1],
//			solverOptions.gridDiameter[2]
//		};
//
//		int3 gridSize =
//		{
//			solverOptions.gridSize[0],
//			solverOptions.gridSize[1],
//			solverOptions.gridSize[2]
//		};
//
//		for (int i = 0; i < lineLength; i++)
//		{
//			d_particles[index].move(dt, gridSize, gridDiameter, t_VelocityField);
//
//			p_VertexBuffer[index_buffer + i].pos.x = d_particles[index].getPosition()->x;
//			p_VertexBuffer[index_buffer + i].pos.y = d_particles[index].getPosition()->y;
//			p_VertexBuffer[index_buffer + i].pos.z = d_particles[index].getPosition()->z;
//
//			switch (solverOptions.colorMode)
//			{
//			case 0: // Velocity
//			{
//				float3* velocity = d_particles[index].getVelocity();
//				double norm = norm3d(velocity->x, velocity->y, velocity->z);
//				p_VertexBuffer[index_buffer + i].colorID.x = norm;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//
//			}
//			case 1: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->x;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			case 2: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->y;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			case 3: // Vx
//			{
//				float velocity = d_particles[index].getVelocity()->z;
//				p_VertexBuffer[index_buffer + i].colorID.x = velocity;
//				p_VertexBuffer[index_buffer + i].colorID.y = index;
//			}
//			}
//
//
//		}
//	}
//
//}
//template <typename T>
//__host__ void PathlineSolver<T>::release()
//{
//	cudaFree(this->d_Particles);
//	cudaFree(this->d_VelocityField);
//	cudaDestroyTextureObject(this->t_VelocityField);
//}
//
template <typename T>
__host__ bool PathlineSolver<T>::solve()
{
	//this->volume_IO.Initialize(this->solverOptions);

	//// TO-DO: Define streamlinesolver for double precision
	//InitializeVelocityField();

	//this->InitializeTexture();

	//this->InitializeParticles();

	//int blockDim = 256;
	//int thread = (this->solverOptions.lines_count / blockDim) + 1;

	//TracingParticles<T> << <blockDim, thread >> > (d_Particles, t_VelocityField, solverOptions, reinterpret_cast<Vertex*>(this->p_VertexBuffer));

	//this->release();

	return true;
}