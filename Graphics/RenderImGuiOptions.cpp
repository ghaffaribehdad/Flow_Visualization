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



	if (ImGui::Combo("Line Rendering Mode", &solverOptions->lineRenderingMode, LineRenderingMode::lineRenderingModeList, LineRenderingMode::lineRenderingMode::COUNT))
	{
		if (this->showStreamlines)
		{
			this->showStreamlines = false;
			this->releaseStreamlines = true;
			this->solverOptions->loadNewfile = false;
		}
		if (this->showStreaklines)
		{
			this->showStreaklines = false;
			this->releaseStreaklines = true;
		}
		if (this->showPathlines)
		{
			this->showPathlines = false;
			this->releasePathlines = true;
		}
	}

	if (ImGui::Checkbox("Compressed Data", &solverOptions->Compressed))
	{


	}
	if (ImGui::InputText("Name", _strdup(solverOptions->outputFileName.c_str()), 100 * sizeof(char)))
	{
	}
	
	if (ImGui::Button("Screenshot"))
	{
		this->saveScreenshot = true;
	}

	if (ImGui::InputInt("Range",&this->screenshotRange,1,10))
	{
	}


	if (ImGui::InputText("File Path", _strdup(solverOptions->filePath.c_str()), 100 * sizeof(char)))
	{
	}

	if (ImGui::InputText("File Name", _strdup(solverOptions->fileName.c_str()), 100*sizeof(char)))
	{
	}

	if (ImGui::Combo("Projection", &solverOptions->projection, Projection::ProjectionList, Projection::Projection::COUNT))
	{
		
	}

	if (ImGui::Checkbox("Periodic", &solverOptions->periodic))
	{

	}




	if (ImGui::InputInt3("Grid Size", solverOptions->gridSize, sizeof(solverOptions->gridSize)))
	{
		this->updateSeedBox = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;
	}
	if (ImGui::InputFloat3("Grid Diameter", solverOptions->gridDiameter, sizeof(solverOptions->gridDiameter)))
	{
		this->updateVolumeBox = true;
		this->updateRaycasting = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}

	if (ImGui::DragFloat3("Velocity Scaling Factor", solverOptions->velocityScalingFactor,0.01f))
	{
		this->updateVolumeBox = true;
		this->updateRaycasting = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;

	}
	if (ImGui::Button("Optical Flow"))
	{
		divideArrayFloat3Int3(solverOptions->velocityScalingFactor, solverOptions->gridDiameter, solverOptions->gridSize);
		this->solverOptions->dt = 1.0f;
		this->updatePathlines = true;
		this->updateStreamlines = true;

	}
	ImGui::SameLine();
	if (ImGui::Button("Reset Scaling Factor"))
	{
		solverOptions->velocityScalingFactor[0] = 1.0f;
		solverOptions->velocityScalingFactor[1] = 1.0f;
		solverOptions->velocityScalingFactor[2] = 1.0f;
		this->updatePathlines = true;
		this->updateStreamlines = true;

	}

	if (ImGui::Combo("Seeding Pattern", (int*)&solverOptions->seedingPattern, SeedPatternList, 3))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;


		if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_GRIDPOINTS)
			solverOptions->lines_count = solverOptions->seedGrid[0] * solverOptions->seedGrid[1] * solverOptions->seedGrid[2];
	}

	if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_GRIDPOINTS)
	{
		if (ImGui::DragInt3("Seed Grid", solverOptions->seedGrid,1,1,1024))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;
			this->updateStreaklines = true;

		}
		solverOptions->lines_count = solverOptions->seedGrid[0] * solverOptions->seedGrid[1] * solverOptions->seedGrid[2];
	}

	if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_TILTED_PLANE)
	{
		if (ImGui::DragInt2("Seed Grid", solverOptions->gridSize_2D, 1, 1, 1024))
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;
			this->updateStreaklines = true;

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
		this->updateStreaklines = true;

	}
	
	if (ImGui::DragFloat3("Seed Box Position", solverOptions->seedBoxPos, 0.01f))
	{
		this->updateSeedBox = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;

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

	if (solverOptions->lineRenderingMode == LineRenderingMode::lineRenderingMode::STREAKLINES || solverOptions->lineRenderingMode == LineRenderingMode::lineRenderingMode::PATHLINES)
	{
		solverOptions->lineLength = solverOptions->lastIdx - solverOptions->firstIdx;
	}

	ImGui::SameLine();

	if (ImGui::InputInt("Last Index", &(solverOptions->lastIdx)))
	{
		this->updatePathlines = true;
		this->updateStreaklines = true;

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
		solverOptions->loadNewfile = true;
		this->solverOptions->fileChanged = true;


		this->raycastingOptions->fileChanged = true;
		this->crossSectionOptions->updateTime = true;

		this->updateCrossSection = true;
		
		this->saved = false;
	}

	ImGui::PopItemWidth();

	if (ImGui::DragFloat("dt", &(solverOptions->dt),0.0001f,0.001f,1.0f,"%.4f"))
	{
		
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;

		
	}

	if (ImGui::InputInt("Lines", &(solverOptions->lines_count)))
	{
		if (this->solverOptions->lines_count <= 0)
		{
			this->solverOptions->lines_count = 1;
		}
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;

	}

	// length of the line is fixed for the pathlines

	if (solverOptions->lineRenderingMode == LineRenderingMode::lineRenderingMode::STREAMLINES)
	{
		if (ImGui::InputInt("Line Length", &(solverOptions->lineLength)))
		{
			this->updateStreamlines = true;
		}
		if (this->solverOptions->lineLength <= 0)
		{
			this->solverOptions->lineLength = 1;
			this->updateStreamlines = true;

		}
	}





	if (ImGui::Combo("Color Mode", &solverOptions->colorMode, ColorMode::ColorModeList, ColorMode::ColorMode::COUNT))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;


	}




	//if (this->streamlineGenerating)
	//{
	//	if (ImGui::InputText("File Path Out", solverOptions->filePath_out, sizeof(solverOptions->filePath_out)))
	//	{
	//	}

	//	if (ImGui::InputText("File Name Out", solverOptions->fileName_out, sizeof(solverOptions->fileName_out)))
	//	{
	//	}
	//}
	
	switch (pauseRendering)
	{
	case(true):
	{
		if (ImGui::Button("Resume", ImVec2(80, 25)))
		{
			pauseRendering = !pauseRendering;
		}
		break;
	}
	case(false):
	{
		if (ImGui::Button("Pause", ImVec2(80, 25)))
		{
			pauseRendering = !pauseRendering;
		}
		break;
	}
	}


	ImGui::SameLine();
	if (ImGui::Button("reset", ImVec2(80, 25)))
	{
		this->updatePathlines = true;
		this->updateStreaklines = true;
	}

	switch (solverOptions->lineRenderingMode)
	{
	case LineRenderingMode::lineRenderingMode::STREAMLINES:
	{
		// Show Lines
		if (this->streamlineRendering || this->streamlineGenerating)
		{
			if (ImGui::Checkbox("Render Streamlines", &this->showStreamlines))
			{
				this->updateStreamlines = true;
				this->solverOptions->loadNewfile = true;


				if (!this->showStreamlines)
				{
					this->releaseStreamlines = true;
					this->solverOptions->loadNewfile = false;
				}
			}


		}
		break;
	}

	case LineRenderingMode::lineRenderingMode::PATHLINES:
	{
		if (this->solverOptions->lastIdx - this->solverOptions->firstIdx >= 2)
		{
			if (ImGui::Checkbox("Render Pathlines", &this->showPathlines))
			{
				this->updatePathlines = true;

				if (!this->showPathlines)
				{
					this->releasePathlines = true;
				}
			}


		}
		break;
	}


	case LineRenderingMode::lineRenderingMode::STREAKLINES:
	{
		if (this->solverOptions->lastIdx - this->solverOptions->firstIdx >= 2)
		{
			if (ImGui::Checkbox("Render Streaklines", &this->showStreaklines))
			{
				this->updateStreaklines = true;

				if (!this->showStreaklines)
				{
					this->releaseStreaklines = true;
				}
			}

		}
		break;
	}


	}




	if (ImGui::Button("Reset View", ImVec2(80, 25)))
	{
		this->camera->SetPosition(-3.91f, 0.05f,-4.94f);
		this->camera->SetLookAtPos({ 0.54f, -0.02f, 0.84f });

		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updatefluctuation = true;
		this->updateFTLE = true;
	}

	ImGui::SameLine();

	if (ImGui::Button("Edge View", ImVec2(80, 25)))
	{
		this->camera->SetPosition(-10.7f, 4.0f, -6.93f);
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
		this->camera->SetPosition(-10.7f, 4.6f, -0.0001f);
		this->camera->SetLookAtPos({ -0.9f, -0.33f, 0 });

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
	

	if (ImGui::InputInt("Realtime time step", &solverOptions->counter))
	{

	}

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

	ImGui::Begin("Rendering Options");


	if(ImGui::Checkbox("Show Seed Box", &renderingOptions->showSeedBox))
	{ }

	if (ImGui::Checkbox("Show Volume Box", &renderingOptions->showVolumeBox))
	{
	}

	if (ImGui::Checkbox("Show Clip Box", &renderingOptions->showClipBox))
	{
	}


	if(ImGui::SliderFloat("Box Radius", &renderingOptions->boxRadius, 0.0f, 0.05f, "%.4f"))
	{
	}

	if (ImGui::Combo("Rendering Mode", &renderingOptions->renderingMode, RenderingMode::RenderingModeList, RenderingMode::RenderingMode::COUNT))
	{
		this->updateShaders = true;
	}

	if (ImGui::Combo("Draw Mode", &renderingOptions->drawMode, DrawMode::DrawModeList, DrawMode::DrawMode::COUNT))
	{
	}


	if (ImGui::InputInt("Line Length", & renderingOptions->lineLength,1,10))
	{
		if (renderingOptions->lineLength < 1)
		{
			renderingOptions->lineLength = 1;
		}
	}

	if (ImGui::ColorEdit4("Background", (float*)&renderingOptions->bgColor))
	{

		this->updateRaycasting = true;
		this->updateDispersion = true;
	}
	

	ImGui::SliderFloat("Tube Radius", &renderingOptions->tubeRadius, 0.0f, 0.02f,"%.5f");            // Edit 1 float using a slider from 0.0f to 1.0f



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
	
	if (ImGui::Checkbox("Identical Data", &this->raycastingOptions->identicalDataset))
	{}


	if (!raycastingOptions->identicalDataset)
	{
		//if (ImGui::InputText("File Path", solverOptions->filePath, sizeof(raycastingOptions->filePath)))
		//{
		//}

		//if (ImGui::InputText("File Name", solverOptions->fileName, sizeof(raycastingOptions->fileName)))
		//{
		//}
	}
	if (ImGui::Checkbox("Enable Raycasintg", &this->showRaycasting)) 
	{
		this->renderingOptions->isRaycasting = this->showRaycasting;
		this->updateRaycasting = true;
	}

	if (ImGui::Combo("Isosurface Measure 0", &raycastingOptions->isoMeasure_0, IsoMeasureModes,(int)IsoMeasure::COUNT))
	{
		this->updateRaycasting = true;
		this->updateTimeSpaceField = true;

	}

	if (ImGui::Button("okay"))
	{

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



	if (ImGui::DragFloat3("Clip Box", raycastingOptions->clipBox, 0.01f))
	{

		this->updateRaycasting = true;
		this->updateTimeSpaceField = true;

	}

	if (ImGui::DragFloat3("Clip Box Center", raycastingOptions->clipBoxCenter, 0.01f))
	{

		this->updateRaycasting = true;
		this->updateTimeSpaceField = true;

	}


	if (ImGui::DragFloat("Isovalue 0", &raycastingOptions->isoValue_0, 0.001f))
	{
		this->updateRaycasting = true;
		this->updateTimeSpaceField = true;
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

	if (ImGui::ColorEdit4("Min Color", (float*)&raycastingOptions->minColor))
	{
		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updateFTLE = true;

	}
	if (ImGui::ColorEdit4("Max Color", (float*)&raycastingOptions->maxColor))
	{
		this->updateRaycasting = true;
		this->updateDispersion = true;
		this->updateFTLE = true;

	}



	if (this->raycastingOptions->fileLoaded)
	{
		ImGui::Text("File is loaded!");
	}
	else
	{
		ImGui::Text("File is not loaded yet!");
	}

	if (raycastingOptions->isoMeasure_0 == IsoMeasure::Velocity_X_Plane ||
		raycastingOptions->isoMeasure_0 == IsoMeasure::Velocity_Y_Plane ||
		raycastingOptions->isoMeasure_0 == IsoMeasure::Velocity_Z_Plane	||
		raycastingOptions->isoMeasure_0 == IsoMeasure::LAMBDA2
		)
	{
		if (ImGui::InputFloat("Min Value", (float*)&raycastingOptions->minVal, 0.001f,0.1f))
		{
			this->updateRaycasting = true;

		}

		if (ImGui::InputFloat("max Value", (float*)&raycastingOptions->maxVal, 0.001f, 0.1f))
		{
			this->updateRaycasting = true;

		}

		if (ImGui::InputFloat("plane Thickness", (float*)& raycastingOptions->planeThinkness, 0.001f, 0.01f))
		{
			this->updateRaycasting = true;
		}

		if (ImGui::InputFloat("Wall-normal clipping", &raycastingOptions->wallNormalClipping, 0.01f, 0.1f))
		{
			if (raycastingOptions->wallNormalClipping > 1.0f)
			{
				raycastingOptions->wallNormalClipping = 1.0f;
			}
			else if (raycastingOptions->wallNormalClipping < 0.0f)
			{
				raycastingOptions->wallNormalClipping = 0.0f;
			}

			this->updateRaycasting = true;
		}
	}

	

	ImGui::End();

}


void RenderImGuiOptions::drawDispersionOptions()
{
	ImGui::Begin("Dispersion Options");


	if (ImGui::Combo("Field Mode", &dispersionOptions->heightMode, heightMode, dispersionOptionsMode::HeightMode::COUNT))
	{

	}

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
	
	if (ImGui::InputInt("Save index", &dispersionOptions->file_counter, 1, 10)) {}


	if (ImGui::InputInt("timeStep", &dispersionOptions->timestep,1,10))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;
	}




	if (ImGui::DragFloat("dt dispersion", &dispersionOptions->dt, 0.0001f,0.001f,1.0f,"%5f"))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;

		this->dispersionOptions->retrace = true;

	}

	if (ImGui::DragFloat("height scale", &dispersionOptions->scale,0.0001f,0.00001f,100.0f,"%5f"))
	{
		this->updateDispersion = true;
		this->updateFTLE = true;
	}

	if (ImGui::Checkbox("Forward-FTLE", &dispersionOptions->forward))
	{
		dispersionOptions->backward = !dispersionOptions->forward;
		this->updateDispersion = true;
		this->updateFTLE = true;

	}

	ImGui::SameLine();
	if (ImGui::Checkbox("Backward-FTLE", &dispersionOptions->backward))
	{
		dispersionOptions->forward = !dispersionOptions->backward;
		this->updateDispersion = true;
		this->updateFTLE = true;

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



void RenderImGuiOptions::drawTimeSpaceOptions()
{
	ImGui::Begin("Time-Space Rendering");

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

	if (ImGui::InputInt("Number of Slices", &fluctuationOptions->streamwiseSlice, 1, 1))
	{
		this->updatefluctuation = true;
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

		solverOptions->timeSteps = solverOptions->lastIdx - solverOptions->firstIdx + 1;

	}

	if (ImGui::InputInt("Last Index", &(fluctuationOptions->lastIdx)))
	{
		if (fluctuationOptions->lastIdx < fluctuationOptions->firstIdx)
		{
			fluctuationOptions->firstIdx = fluctuationOptions->lastIdx;
		}

		solverOptions->timeSteps = solverOptions->lastIdx - solverOptions->firstIdx + 1;

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



void RenderImGuiOptions::drawDataset()
{

	ImGui::Begin("Datasets");

	if (!raycastingOptions->identicalDataset)
	{

		if (ImGui::Combo("RaycastingDataset", reinterpret_cast<int*>(&this->raycastyingDataset), Dataset::datasetList, Dataset::Dataset::COUNT))
		{
			//switch (raycastyingDataset)
			//{

			//}
	
		}
	}

	if (ImGui::Combo("Dataset", reinterpret_cast<int*>(&this->dataset),Dataset::datasetList, Dataset::Dataset::COUNT))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;
		this->updateRaycasting = true;

		this->solverOptions->fileChanged = true;
		solverOptions->loadNewfile = true;
		this->solverOptions->fileChanged = true;


		switch (dataset)
		{

			case Dataset::Dataset::KIT2REF_COMP:
			{

				
				this->solverOptions->fileName = "FieldComp";
				this->solverOptions->filePath = "G:\\KIT2Padded\\OscillatingWall\\Compressed\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 900;
				this->solverOptions->Compressed = true;
				break;			
			}

			case Dataset::Dataset::KIT2OW:
			{
				this->solverOptions->fileName = "FieldP";
				this->solverOptions->filePath = "G:\\KIT2Padded\\OscillatingWall\\Padded\\";
				
				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				break;			
			}
			case Dataset::Dataset::KIT2BF:
			{
				this->solverOptions->fileName = "FieldP";
				this->solverOptions->filePath = "G:\\KIT2Padded\\VirtualBody\\Padded\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				break;			
			}
			case Dataset::Dataset::KIT3_FLUC:
			{
				this->solverOptions->fileName = "FieldP";
				this->solverOptions->filePath = "F:\\Dataset\\KIT3\\binary_fluc_z_major\\Padded\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64,503,2048);



				this->solverOptions->dt = 0.001f;
				break;

			}

			case Dataset::Dataset::KIT3_COMPRESSED:
			{
				this->solverOptions->fileName = "Fluc_Comp_";
				this->solverOptions->filePath = "G:\\KIT3\\Comp_Fluc\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 64000000;
				break;

			}


			case Dataset::Dataset::KIT3_OF_AVG50_COMPRESSED:
			{
				this->solverOptions->fileName = "OF_AVG_COMP_50_";
				this->solverOptions->filePath = "G:\\KIT3\\Comp_OF_AVG50\\";


				this->solverOptions->firstIdx = 551;
				this->solverOptions->lastIdx = 949;
				this->solverOptions->currentIdx = 551;

				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);


				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 96000000;

				break;

			}

			case Dataset::Dataset::KIT3_OF_COMPRESSED_FAST:
			{
				this->solverOptions->fileName = "OF_AVG_COMP_";
				this->solverOptions->filePath = "G:\\KIT3\\Comp_OF_Fast\\";


				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 949;
				this->solverOptions->currentIdx = 500;

				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);


				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 96000000;

				break;

			}

			case Dataset::Dataset::KIT3_AVG_COMPRESSED_10:
			{
				this->solverOptions->fileName = "FieldTimeAVG_";
				this->solverOptions->filePath = "G:\\KIT3TimeAvg10\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 900;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 70000000;

				break;

			}

			case Dataset::Dataset::KIT3_AVG_COMPRESSED_50:
			{
				this->solverOptions->fileName = "Field_AVG_Comp_";
				this->solverOptions->filePath = "G:\\KIT3\\Comp_TimeAVG50\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 900;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 70000000;

				break;

			}

			case Dataset::Dataset::KIT3_SECONDARY_COMPRESSED:
			{
				this->solverOptions->fileName = "FieldSecondary";
				this->solverOptions->filePath = "G:\\KIT3Secondary\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 900;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = true;
				this->solverOptions->maxSize = 70000000;

				break;

			}

			case Dataset::Dataset::KIT3_OF:
			{
				this->solverOptions->fileName = "OF_m_stream";
				this->solverOptions->filePath = "F:\\KIT3OpticalFlowPeriodic\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);
							   
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;


				break;

			}



			case Dataset::Dataset::KIT3_TIME_SPACE_1000_TZY:
			{
				this->solverOptions->fileName = "streak_3D_fluc_spanTime_1000_tzy_part_";
				this->solverOptions->filePath = "G:\\KIT3TimeSpaceStreak\\";



				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 2.0f, 1.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 2.0f, 1.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 2.0f, 1.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1000, 1024, 250);



				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = false;

				break;

			}

			case Dataset::Dataset::KIT3_TIME_SPACE_1000_TYX:
			{
				this->solverOptions->fileName = "streak_fluctuation_spanTime_1000_TYX";
				this->solverOptions->filePath = "G:\\KIT3TimeSpaceStreak\\";



				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 2.0f, 1.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 2.0f, 1.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 2.0f, 1.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1000, 503, 64);



				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = false;

				break;

			}

			case Dataset::Dataset::RBC:
			{
				this->solverOptions->fileName = "Field";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\tui_ra1e5\\";



				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 1.0f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);



				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->Compressed = false;

				break;

			}


			case Dataset::Dataset::RBC_AVG:
			{
				this->solverOptions->fileName = "Field_AVG";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\timeAVG\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 1.0f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);


				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->Compressed = false;

				break;

			}

			case Dataset::Dataset::RBC_OF:
			{
				this->solverOptions->fileName = "OF_temperature";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\OF\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 1.0f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);


				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->Compressed = false;

				break;

			}


			case Dataset::Dataset::RBC_AVG_OF_600:
			{
				this->solverOptions->fileName = "Field_OF_AVG600_";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\timeAVG\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 1.0f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);


				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->Compressed = false;

				break;

			}

			case Dataset::Dataset::RBC_AVG500:
			{
				this->solverOptions->fileName = "Field_AVG500_";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\timeAVG\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 1.0f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 1.0f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);


				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->Compressed = false;

				break;

			}
			case Dataset::Dataset::TEST_FIELD:
			{
				this->solverOptions->fileName = "Field";
				this->solverOptions->filePath = "D:\\VelocityFieldGen\\";
				this->solverOptions->gridDiameter[0] = 7.854f;
				this->solverOptions->gridDiameter[1] = 2.0f;
				this->solverOptions->gridDiameter[2] = 3.1415f;

				this->solverOptions->seedBox[0] = 7.854f;
				this->solverOptions->seedBox[1] = 2.0f;
				this->solverOptions->seedBox[2] = 3.1415f;

				this->solverOptions->gridSize[0] = 64;
				this->solverOptions->gridSize[1] = 503;
				this->solverOptions->gridSize[2] = 2048;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->Compressed = false;

				break;

			}


		}
	}


	ImGui::End();

}





void RenderImGuiOptions::drawTimeSpaceField()
{
	ImGui::Begin("Time-Space");


	// Show Cross Sections
	if (ImGui::Checkbox("Render Time Space", &this->showTimeSpaceField))
	{
		this->updateTimeSpaceField = true;
	}


	if (ImGui::DragFloat("Isovalue", &timeSpace3DOptions->isoValue, 1.0f))
	{
		this->updateTimeSpaceField = true;
	}


	if (ImGui::DragFloat("Sampling Rate", &timeSpace3DOptions->samplingRate, 0.00001f, 0.0001f, 1.0f, "%.5f"))
	{
		if (timeSpace3DOptions->samplingRate < 0.0001f)
		{
			timeSpace3DOptions->samplingRate = 0.0001f;
		}

		this->updateTimeSpaceField = true;

	}

	if (ImGui::DragFloat("Tolerance", &timeSpace3DOptions->tolerance, 0.00001f, 0.0001f, 5, "%5f"))
	{
		this->updateTimeSpaceField = true;

	}


	if (ImGui::DragFloat("Isovalue Tolerance", &timeSpace3DOptions->isoValueTolerance,0.001f,0.01f,100, "%5f"))
	{
		this->updateTimeSpaceField = true;

	}

	if (ImGui::DragInt("Iterations", &timeSpace3DOptions->iteration,1,10,100))
	{
		this->updateTimeSpaceField = true;

	}

	if (ImGui::InputInt("First time step", &timeSpace3DOptions->t_first))
	{

	}

	if (ImGui::InputInt("Last time step", &timeSpace3DOptions->t_last))
	{

	}

	if(ImGui::InputFloat("Wall-normal clipping", &timeSpace3DOptions->wallNormalClipping, 0.01f, 0.1f))
	{
		if (timeSpace3DOptions->wallNormalClipping > 1.0f)
		{
			timeSpace3DOptions->wallNormalClipping = 1.0f;
		}
		else if (timeSpace3DOptions->wallNormalClipping < 0.0f)
		{
			timeSpace3DOptions->wallNormalClipping = 0.0f;
		}

		this->updateTimeSpaceField = true;
	}

	if (ImGui::ColorEdit3("Isosurface Color", (float*)& timeSpace3DOptions->color))
	{

		this->updateTimeSpaceField = true;

	}



	if (ImGui::InputFloat("Min Value", (float*)&timeSpace3DOptions->minVal, 1.0f))
	{
		this->updateTimeSpaceField = true;

	}

	if (ImGui::InputFloat("max Value", (float*)&timeSpace3DOptions->maxVal, 1.0f))
	{
		this->updateTimeSpaceField = true;

	}


	ImGui::End();
}