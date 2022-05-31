#pragma once
#include "../Raycaster/Raycasting.h"
#include "../Cuda/CudaArray.h"
#include "../Cuda/cudaSurface.h"


class VisitingMap : public Raycasting
{

public:

	bool Initialization();

private:

	CudaSurface s_visitationMap;
	CudaArray_3D<float4> a_visitationMap;

};