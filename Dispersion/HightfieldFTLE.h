#pragma once

#include "DispersionTracer.h"
#include "ftleHelperFunctions.h"



class HeightfieldFTLE : public HeightfieldGenerator
{


public:


private:

	virtual bool InitializeParticles() override;
	virtual void trace3D_path_Single() override;

};