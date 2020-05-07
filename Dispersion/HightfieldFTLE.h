#pragma once

#include "DispersionTracer.h"
#include "ftleHelperFunctions.h"
#include "..//Options/FSLEOptions.h"



class HeightfieldFTLE : public HeightfieldGenerator
{


public:

	FSLEOptions * fsleOptions = nullptr;

	virtual void show(RenderImGuiOptions* renderImGuiOptions)
	{
		if (renderImGuiOptions->showFTLE)
		{
			if (this->dispersionOptions->retrace)
			{
				//this->dispersionTracer.retrace();
				//this->dispersionOptions.retrace = false;
			}
			if (!this->dispersionOptions->initialized)
			{

				this->initialize(cudaAddressModeWrap, cudaAddressModeBorder, cudaAddressModeWrap);
				this->dispersionOptions->initialized = true;
			}
			// Override draw
			this->draw();

			if (renderImGuiOptions->updateFTLE)
			{
				this->updateScene();
				renderImGuiOptions->updateFTLE = false;

			}
		}
		else
		{
			if (dispersionOptions->released)
			{
				this->release();
				dispersionOptions->released = false;
			}
		}
	}

	
private:

	virtual bool InitializeParticles() override;
	virtual void trace3D_path_Single() override;
	virtual void rendering() override;
	virtual bool singleSurfaceInitialization() override;
	void gradient3D_Single_ftle();
	void correlation();
	virtual bool initializeBoundingBox() override;

	float * h_pearson;

	float * h_ftle;
	float * h_height;
	float * d_mean_ftle;
	float * d_ftle;
	float * d_height;
	float * d_mean_height;
	float * d_pearson_cov;
	float * d_pearson_var_ftle;
	float * d_pearson_var_height;
};