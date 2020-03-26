#include "RenderImGuiOptions.h"
#include "..//Raycaster/Raycasting_Helper.h"
#include "..//Cuda/Cuda_helper_math_host.h"


void RenderImGuiOptions::drawSolverOptions()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();


	// Solver Options
	ImGui::Begin("Solver Options");

	ImGui::Text("Mode: ");
	ImGui::SameLine();

	//Solver Mode

	if (ImGui::Checkbox("Streamline", &this->streamlineRendering))
	{
		this->pathlineRendering = !this->streamlineRendering;
		this->streamlineGenerating = !this->streamlineRendering;
	}

	ImGui::SameLine();
	if (ImGui::Checkbox("Pathline", &this->pathlineRendering))
	{
		this->streamlineGenerating = !this->pathlineRendering;
		this->streamlineRendering = !this->pathlineRendering;

	}

	if (ImGui::Checkbox("Streamline Gen.", &this->streamlineGenerating))
	{
		this->streamlineRendering = !this->streamlineGenerating;
		this->pathlineRendering = !this->streamlineGenerating;
		this->solverOptions->lines_count = 1000;
		this->solverOptions->lineLength = 1000;

	}


	
	if (ImGui::InputText("File Path", solverOptions->filePath, sizeof(solverOptions->filePath)))
	{
	}

	if (ImGui::InputText("File Name", solverOptions->fileName, sizeof(solverOptions->fileName)))
	{
	}

	if (ImGui::Combo("Projection", &solverOptions->projection, ProjectionList, 4))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}

	if (ImGui::Checkbox("Periodic", &solverOptions->periodic))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}


	if (ImGui::InputInt3("Grid Size", solverOptions->gridSize, sizeof(solverOptions->gridSize)))
	{
		this->updateSeedBox = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}
	if (ImGui::InputFloat3("Grid Diameter", solverOptions->gridDiameter, sizeof(solverOptions->gridDiameter)))
	{
		this->updateVolumeBox = true;
		this->updateRaycasting = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}

	if (ImGui::Combo("Seeding Pattern", (int*)&solverOptions->seedingPattern, SeedPatternList, 3))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;

		if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_GRIDPOINTS)
			solverOptions->lines_count = solverOptions->seedGrid[0] * solverOptions->seedGrid[1] * solverOptions->seedGrid[2];
	}

	if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_GRIDPOINTS)
	{
		if (ImGui::DragInt3("Seed Grid", solverOptions->seedGrid,1,1,1024))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;
		}
		solverOptions->lines_count = solverOptions->seedGrid[0] * solverOptions->seedGrid[1] * solverOptions->seedGrid[2];
	}

	if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_TILTED_PLANE)
	{
		if (ImGui::DragInt2("Seed Grid", solverOptions->gridSize_2D, 1, 1, 1024))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;
		}

		if (ImGui::DragFloat("Wall-normal Distance", &solverOptions->seedWallNormalDist, 0.001f, 0.0, solverOptions->gridDiameter[1], "%4f"))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;

		}

		if (ImGui::DragFloat("Tilt Deg", &solverOptions->tilt_deg, 0.1f, 0.0, 45.0f))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;

		}


		solverOptions->lines_count = solverOptions->gridSize_2D[0] * solverOptions->gridSize_2D[1];
	}



	if (ImGui::DragFloat3("Seed Box", solverOptions->seedBox, 0.01f))
	{
		this->updateSeedBox = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}
	
	if (ImGui::DragFloat3("Seed Box Position", solverOptions->seedBoxPos, 0.01f))
	{
		this->updateSeedBox = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}


	ImGui::PushItemWidth(75);
	if (ImGui::InputInt("First Index", &(solverOptions->firstIdx)))
	{
		if (solverOptions->lastIdx < solverOptions->firstIdx)
		{
			solverOptions->firstIdx = solverOptions->lastIdx;
		}
		this->updatePathlines = true;
		if (solverOptions->currentIdx < solverOptions->firstIdx)
		{
			solverOptions->currentIdx = solverOptions->firstIdx;
			this->updateStreamlines = true;

		}

	}

	ImGui::SameLine();

	if (ImGui::InputInt("Last Index", &(solverOptions->lastIdx)))
	{
		this->updatePathlines = true;

	}

	if (ImGui::InputInt("Current Index", &(solverOptions->currentIdx),1,2))
	{
		if (solverOptions->currentIdx < solverOptions->firstIdx)
		{
			solverOptions->currentIdx = solverOptions->firstIdx;
		}

		if (solverOptions->currentIdx > solverOptions->lastIdx)
		{
			solverOptions->currentIdx = solverOptions->lastIdx;
		}

		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->crossSectionOptions->updateTime = true;
		this->updateCrossSection = true;
	}

	ImGui::PopItemWidth();

	if (ImGui::DragFloat("dt", &(solverOptions->dt),0.0001f,0.001f,1.0f,"%.4f"))
	{
		
		this->updateStreamlines = true;
		this->updatePathlines = true;
		
	}

	if (ImGui::InputInt("Lines", &(solverOptions->lines_count)))
	{
		if (this->solverOptions->lines_count <= 0)
		{
			this->solverOptions->lines_count = 1;
		}
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}

	// length of the line is fixed for the pathlines
	if (this->streamlineRendering || this->streamlineGenerating)
	{
		if (ImGui::InputInt("Line Length", &(solverOptions->lineLength)))
		{
			if (this->solverOptions->lineLength <= 0)
			{
				this->solverOptions->lineLength = 1;
			}
			this->updateStreamlines = true;
			this->updatePathlines = true;

		}
	}




	if (ImGui::Combo("Color Mode", &solverOptions->colorMode, ColorModeList, 7))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;

	}




	if (this->streamlineGenerating)
	{
		if (ImGui::InputText("File Path Out", solverOptions->filePath_out, sizeof(solverOptions->filePath_out)))
		{
		}

		if (ImGui::InputText("File Name Out", solverOptions->fileName_out, sizeof(solverOptions->fileName_out)))
		{
		}
	}



	// Show Lines
	if (this->streamlineRendering || this->streamlineGenerating)
	{
		if (ImGui::Checkbox("Render Streamlines", &this->showStreamlines))
		{
			this->updateStreamlines = true;
		}
	}


	else // PathlineRendering
	{
		if (this->solverOptions->lastIdx - this->solverOptions->firstIdx >= 2)
		{
			if (ImGui::Checkbox("Render Pathlines", &this->showPathlines))
			{
				this->updatePathlines = true;

			}
		}

	}

	if (ImGui::Button("Reset View", ImVec2(80, 25)))
	{
		this->camera->SetPosition(0, 5, -10);
		this->camera->SetLookAtPos({ 0, 0, 0 });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;
	}

	ImGui::SameLine();

	if (ImGui::Button("Edge View", ImVec2(80, 25)))
	{
		this->camera->SetPosition(-10.7f, 4.0f, -5.37f);
		this->camera->SetLookAtPos({ 0.75f,-0.35f,0.55f });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;

	}
	ImGui::SameLine();

	if (ImGui::Button("Top", ImVec2(80, 25)))
	{
		this->camera->SetPosition(0, 0, +10);
		this->camera->SetLookAtPos({ 0, 0, 0 });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;

	}


	if (ImGui::Button("Bottom", ImVec2(80, 25)))
	{
		this->camera->SetPosition(0, 0, -10);
		this->camera->SetLookAtPos({ 0, 0, 0 });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;

	}
	ImGui::SameLine();
	if (ImGui::Button("Side", ImVec2(80, 25)))
	{
		this->camera->SetPosition(5, 5, 10);
		this->camera->SetLookAtPos({ 0, 0, 0 });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;

	}

	ImGui::SameLine();
	if (ImGui::Button("b2f", ImVec2(80, 25)))
	{
		this->camera->SetPosition(-10.7, 4.6, -0.0001);
		this->camera->SetLookAtPos({ -0.9, -0.33, 0 });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;

	}



	ImGui::End();

}


void RenderImGuiOptions::drawLog()
{
	ImGui::Begin("Log");



	// calculatin FPS
	static int fpsCounter = 0;
	static std::string fpsString = "FPS : 0";
	fpsCounter += 1;

	static float fps_array[10];
	static int fps_arrayCounter = 0;

	if (fpsTimer->GetMilisecondsElapsed() > 100.0)
	{
		fpsString = "FPS: " + std::to_string(10 * fpsCounter);

		if (fps_arrayCounter < 10)
		{
			fps_array[fps_arrayCounter] = static_cast<float>(fpsCounter / 10.0f);
			fps_arrayCounter++;
		}
		else
		{
			fps_arrayCounter = 0;
		}

		fpsCounter = 0;
		fpsTimer->Restart();
	}

	strcpy_s(this->log, sizeof(fpsString), fpsString.c_str());

	ImGui::PlotLines("Frame Times", fps_array, IM_ARRAYSIZE(fps_array), 0, NULL, 0, 50, ImVec2(250, 80));

	if (ImGui::InputTextMultiline("Frame per Second", this->log, 1000, ImVec2(100, 35)))
	{

	}

	eyePos[0] = XMFloat3ToFloat3(camera->GetPositionFloat3()).x;
	eyePos[1] = XMFloat3ToFloat3(camera->GetPositionFloat3()).y;
	eyePos[2] = XMFloat3ToFloat3(camera->GetPositionFloat3()).z;


	viewDir[0] = XMFloat3ToFloat3(camera->GetViewVector()).x;
	viewDir[1] = XMFloat3ToFloat3(camera->GetViewVector()).y;
	viewDir[2] = XMFloat3ToFloat3(camera->GetViewVector()).z;

	upDir[0] = XMFloat3ToFloat3(camera->GetUpVector()).x;
	upDir[1] = XMFloat3ToFloat3(camera->GetUpVector()).y;
	upDir[2] = XMFloat3ToFloat3(camera->GetUpVector()).z;



	ImGui::InputFloat3("eye Position", this->eyePos, 2);
	ImGui::InputFloat3("View Dirrection", this->viewDir, 2);
	ImGui::InputFloat3("Up Vector", this->upDir, 2);



	ImGui::End();
}




void RenderImGuiOptions::render()
{

	//Assemble Together Draw Data
	ImGui::Render();

	//Render Draw Data
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());


}


void RenderImGuiOptions::drawLineRenderingOptions()
{

	ImGui::Begin("Line Rendering Options");


	if(ImGui::Checkbox("Show Seed Box", &renderingOptions->showSeedBox))
	{ }

	if (ImGui::Checkbox("Show Volume Box", &renderingOptions->showVolumeBox))
	{
	}
	


	if (ImGui::ColorEdit3("Background", (float*)&bgColor))
	{
		renderingOptions->bgColor[0] = bgColor[0];
		renderingOptions->bgColor[1] = bgColor[1];
		renderingOptions->bgColor[2] = bgColor[2];


		this->updateRaycasting = true;
		this->updateDispersion = true;
	}
	

	ImGui::SliderFloat("Tube Radius", &renderingOptions->tubeRadius, 0.0f, 0.02f,"%.4f");            // Edit 1 float using a slider from 0.0f to 1.0f



	ImGui::Text("Color Coding:");

	if (ImGui::ColorEdit4("Minimum", (float*)&renderingOptions->minColor))
	{

	}
	if (ImGui::InputFloat("Min Value", (float*)& renderingOptions->minMeasure, 0.1f)) 
	{

	}

	if (ImGui::ColorEdit4("Maximum", (float*)&renderingOptions->maxColor))
	{

	}
	if (ImGui::InputFloat("Max Value", (float*)& renderingOptions->maxMeasure, 0.1f)) 
	{

	}




	ImGui::End();

}

void RenderImGuiOptions::drawRaycastingOptions()
{


	ImGui::Begin("Raycasting Options");
	


	if (ImGui::Checkbox("Enable Raycasintg", &this->showRaycasting)) 
	{
		this->renderingOptions->isRaycasting = this->showRaycasting;
		this->updateRaycasting = true;
	}

	if (ImGui::Combo("Isosurface Measure 0", &raycastingOptions->isoMeasure_0, IsoMeasureModes,(int)IsoMeasure::COUNT))
	{
		this->updateRaycasting = true;
	}


	if (ImGui::DragFloat("Sampling Rate 0", &raycastingOptions->samplingRate_0, 0.00001f,0.0001f,1.0f,"%.5f"))
	{
		if (raycastingOptions->samplingRate_0 < 0.0001f)
		{
			raycastingOptions->samplingRate_0 = 0.0001f;
		}
		this->updateRaycasting	= true;
		this->updateDispersion	= true;
		this->updateFTLE = true;


	}

	if (ImGui::DragFloat("Isovalue 0", &raycastingOptions->isoValue_0, 0.001f))
	{
		this->updateRaycasting = true;
	}

	if (ImGui::DragFloat("Tolerance 0", &raycastingOptions->tolerance_0, 0.00001f,0.0001f,5,"%5f"))
	{
		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updateFTLE = true;

	}



	ImGui::Text("Surfaces color 0:");


	if (ImGui::ColorEdit3("Isosurface Color 0", (float*)& raycastingOptions->color_0))
	{
		this->updateRaycasting = true;
		this->updateDispersion = true;

	}

	if (this->raycastingOptions->fileLoaded)
	{
		ImGui::Text("File is loaded!");
	}
	else
	{
		ImGui::Text("File is not loaded yet!");
	}

	ImGui::End();

}


void RenderImGuiOptions::drawDispersionOptions()
{
	ImGui::Begin("Dispersion Options");

	if (solverOptions->lastIdx - solverOptions->firstIdx > 0)
	{
		if (ImGui::Checkbox("Enable Terrain Rendering", &this->showDispersion))
		{
			this->renderingOptions->isRaycasting = this->showDispersion;
			this->updateDispersion = true;
			this->dispersionOptions->released = false;
		}
	}


	if (solverOptions->lastIdx - solverOptions->firstIdx > 0)
	{
		if (ImGui::Checkbox("Enable FTLE rendering", &this->showFTLE))
		{
			this->renderingOptions->isRaycasting = this->showFTLE;
			this->updateFTLE = true;
			this->dispersionOptions->released = false;
		}
	}

	if (ImGui::Combo("Rendering Mode", &dispersionOptions->renderingMode, HeightfieldRenderingMode, 2)) {}


	if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::DOUBLE_SURFACE)
	{
		ImGui::Separator();
		ImGui::Text("Secondary File:");
		if (ImGui::InputText("File Path Secondary", dispersionOptions->filePathSecondary, sizeof(dispersionOptions->filePathSecondary)))
		{
		}

		if (ImGui::InputText("File Name Secondary", dispersionOptions->fileNameSecondary, sizeof(dispersionOptions->fileNameSecondary)))
		{
		}
		ImGui::Separator();
	}
	
	if (ImGui::InputInt("Save index", &dispersionOptions->file_counter, 1, 10)) {}


	if (ImGui::InputInt("timeStep", &dispersionOptions->timestep,1,10))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;

		//this->dispersionOptions->saveScreenshot = true;
	}




	if (ImGui::DragFloat("dt dispersion", &dispersionOptions->dt, 0.0001f,0.001f,1.0f,"%5f"))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;

		this->dispersionOptions->retrace = true;

	}


	if (ImGui::DragFloat("ftle Isosurface", &dispersionOptions->ftleIsoValue, 0.0001f, 0.0f, 100.0f, "%5f"))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;

	}


	if (ImGui::DragFloat("Wall-normal Distance", &dispersionOptions->seedWallNormalDist,0.001f,0.0,solverOptions->gridDiameter[1],"%4f"))
	{
		this->updateDispersion = true;
		this->dispersionOptions->retrace = true;

	}

	if (ImGui::DragFloat("Tilt Deg", &dispersionOptions->tilt_deg, 0.1f, 0.0, 45.0f))
	{
		this->updateDispersion = true;
		this->dispersionOptions->retrace = true;

	}


	if (ImGui::DragFloat("Height Tolerance", &dispersionOptions->hegiht_tolerance,0.0001f,0.0001f,1,"%8f"))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;

	}


	if (!this->showFTLE)
	{
		if (ImGui::Combo("Color Coding", &dispersionOptions->colorCode, ColorCode_DispersionList, 9))
		{
			this->updateDispersion = true;

		}
	}
	

	

	if (dispersionOptions->renderingMode == dispersionOptionsMode::HeightfieldRenderingMode::SINGLE_SURFACE)
	{

		ImGui::Text("Color Coding:");

		if (ImGui::ColorEdit4("Minimum", (float*)&dispersionOptions->minColor))
		{
			updateDispersion = true;
			this->updateFTLE = true;

		}
		if (ImGui::InputFloat("Min Value", (float*)&dispersionOptions->min_val, 0.1f))
		{
			updateDispersion = true;
			this->updateFTLE = true;

		}

		if (ImGui::ColorEdit4("Maximum", (float*)&dispersionOptions->maxColor))
		{
			updateDispersion = true;
			this->updateFTLE = true;

		}

		if (ImGui::InputFloat("Max Value", (float*)&dispersionOptions->max_val, 0.1f))
		{
			updateDispersion = true;
			this->updateFTLE = true;

		}
	}
	else
	{
		if (ImGui::DragFloat("Transparency Secondary", &dispersionOptions->transparencySecondary, 0.001f, 0.0f, 1.0f))
		{
			this->updateDispersion = true;
		}
	}

	if (ImGui::InputInt2("Grid Size 2D", dispersionOptions->gridSize_2D, sizeof(dispersionOptions->gridSize_2D)))
	{
		if (dispersionOptions->gridSize_2D[0] <= 0)
		{
			dispersionOptions->gridSize_2D[0] = 2;
		}

		if (dispersionOptions->gridSize_2D[1] <= 0)
		{
			dispersionOptions->gridSize_2D[1] = 2;
		}
		//this->dispersionOptions->retrace = true;
		this->updateDispersion = true;

	}

	if (ImGui::Checkbox("Enable Marching", &dispersionOptions->marching))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;
	}

	
	ImGui::End();

}



void RenderImGuiOptions::drawFluctuationHeightfieldOptions()
{
	ImGui::Begin("Fluctuation Height-field");

	if (solverOptions->lastIdx - solverOptions->firstIdx > 0)
	{
		if (ImGui::Checkbox("Enable Rendering", &this->showFluctuationHeightfield))
		{
			this->renderingOptions->isRaycasting = this->showFluctuationHeightfield;
			this->updatefluctuation = true;
		}
	}

	if (ImGui::Combo("Field Mode", &fluctuationOptions->fieldMode, FieldMode, 3))
	{

		if (ImGui::InputText("File Path", fluctuationOptions->filePath, sizeof(fluctuationOptions->filePath)))
		{
		}

		if (ImGui::InputText("File Name", fluctuationOptions->fileName, sizeof(fluctuationOptions->fileName)))
		{
		}


	}



	if (ImGui::DragFloat("Height Tolerance", &fluctuationOptions->hegiht_tolerance, 0.0001f, 0.0001f, 1, "%8f"))
	{
		this->updatefluctuation = true;
	}



	ImGui::Text("Color Coding:");

	if (ImGui::ColorEdit4("Minimum", (float*)& fluctuationOptions->minColor))
	{
		updatefluctuation = true;
	}
	if (ImGui::InputFloat("Min Value", (float*)& fluctuationOptions->min_val, 0.1f))
	{
		updatefluctuation = true;
	}

	if (ImGui::ColorEdit4("Maximum", (float*)& fluctuationOptions->maxColor))
	{
		updatefluctuation = true;
	}

	if (ImGui::InputFloat("Max Value", (float*)& fluctuationOptions->max_val, 0.1f))
	{
		updatefluctuation = true;
	}

	ImGui::Separator();

	if (ImGui::DragInt("wall-normal Size", &fluctuationOptions->wallNormalgridSize,1,1,solverOptions->gridSize[1]))
	{

	}

	if (ImGui::InputInt("wall-normal", &fluctuationOptions->wallNoramlPos,1,5))
	{
		if (fluctuationOptions->wallNoramlPos > fluctuationOptions->wallNormalgridSize)
		{
			fluctuationOptions->wallNoramlPos = fluctuationOptions->wallNormalgridSize;
		}
		this->updatefluctuation = true;


	}


	if (ImGui::Checkbox("Absolute Value", &fluctuationOptions->usingAbsolute))
	{
		this->updatefluctuation = true;
	}


	if (ImGui::DragFloat("height scale", &fluctuationOptions->height_scale,0.01f,0,10.0f))
	{
		this->updatefluctuation = true;

	}

	if (ImGui::DragFloat("height offset", &fluctuationOptions->offset, 0.01f, 0, 10.0f))
	{
		this->updatefluctuation = true;

	}
	
	if (ImGui::DragFloat("height clamp", &fluctuationOptions->heightLimit, 0.01f, 0, 5.0f))
	{
		this->updatefluctuation = true;

	}


	ImGui::Separator();

	// Solver Options
	if (ImGui::InputText("File Path", fluctuationOptions->filePath, sizeof(fluctuationOptions->filePath)))
	{
	}

	if (ImGui::InputText("File Name", fluctuationOptions->fileName, sizeof(fluctuationOptions->fileName)))
	{
	}



	if (ImGui::InputInt("First Index", &(fluctuationOptions->firstIdx)))
	{
		if (fluctuationOptions->lastIdx < fluctuationOptions->firstIdx)
		{
			fluctuationOptions->firstIdx = fluctuationOptions->lastIdx;
		}
	}

	if (ImGui::InputInt("Last Index", &(fluctuationOptions->lastIdx)))
	{
		if (fluctuationOptions->lastIdx < fluctuationOptions->firstIdx)
		{
			fluctuationOptions->firstIdx = fluctuationOptions->lastIdx;
		}

	}


	if (ImGui::DragFloat("Sampling Rate 0", &fluctuationOptions->samplingRate_0, 0.00001f, 0.00001f, 1.0f, "%.5f"))
	{
		if (fluctuationOptions->samplingRate_0 < 0.0001f)
		{
			fluctuationOptions->samplingRate_0 = 0.0001f;
		}

		this->updatefluctuation = true;
	}

	ImGui::End();

}






void RenderImGuiOptions::drawCrossSectionOptions()
{
	ImGui::Begin("Cross-Section");


	// Show Cross Sections
	if (ImGui::Checkbox("Render Cross Section", &this->showCrossSection))
	{
		this->updateCrossSection = true;
	}

	if (ImGui::Checkbox("Filter Extremum", &this->crossSectionOptions->filterMinMax))
	{
		this->updateCrossSection = true;
	}


	if (ImGui::Combo("Cross Section Mode",reinterpret_cast<int*>(&crossSectionOptions->mode), spanMode, 3))
	{

	}

	if (ImGui::InputInt("slice", &crossSectionOptions->slice,1,10)) 
	{
		this->updateCrossSection = true;


		switch (crossSectionOptions->crossSectionMode)
		{
			case CrossSectionOptionsMode::CrossSectionMode::XY_SECTION:
			{
				break;
			}
			case CrossSectionOptionsMode::CrossSectionMode::XZ_SECTION:
			{
				break;
			}
			case CrossSectionOptionsMode::CrossSectionMode::YZ_SECTION:
			{
				break;
			}
		}
	}


	ImGui::Text("Color Coding:");

	if (ImGui::ColorEdit4("Minimum", (float*)&crossSectionOptions->minColor))
	{

		this->updateCrossSection = true;

	}
	if (ImGui::InputFloat("Min Value", (float*)&crossSectionOptions->min_val, 0.1f))
	{
		this->updateCrossSection = true;

	}

	if (ImGui::ColorEdit4("Maximum", (float*)&crossSectionOptions->maxColor))
	{
		this->updateCrossSection = true;

	}
	if (ImGui::InputFloat("Max Value", (float*)&crossSectionOptions->max_val, 0.1f))
	{
		this->updateCrossSection = true;

	}


	if (ImGui::DragFloat("Sampling Rate", &crossSectionOptions->samplingRate, 0.00001f, 0.0001f, 1.0f, "%.5f"))
	{
		if (crossSectionOptions->samplingRate < 0.0001f)
		{
			crossSectionOptions->samplingRate = 0.0001f;
		}
		this->updateCrossSection = true;

	}

	if (ImGui::DragFloat("min/max threshold", &crossSectionOptions->min_max_threshold,0.0001f, 0.0001f, 10.0f, "%.5f"))
	{
		this->updateCrossSection = true;
	}


	if (ImGui::Button("Save Screenshot", ImVec2(80, 25)))
	{
		this->crossSectionOptions->saveScreenshot = true;

	}


	ImGui::End();
}





void RenderImGuiOptions::drawTurbulentMixingOptions()
{

	ImGui::Begin("Turbulent Mixing");


	// Show Cross Sections
	if (ImGui::Checkbox("Show Mixing", &this->showTurbulentMixing))
	{
		if (this->turbulentMixingOptions->initialized)
		{
			this->releaseTurbulentMixing = true;
		}
	}









	ImGui::End();

}
