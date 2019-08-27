#include "RenderImGuiOptions.h"

void RenderImGuiOptions::drawSolverOptions()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Solver Options");

	ImGui::Text("Mode: ");
	ImGui::SameLine();

	//Solver Mode

	if (ImGui::Checkbox("Streamline", &this->streamlineRendering))
	{
		this->pathlineRendering = !this->streamlineRendering;
	}
	ImGui::SameLine();
	if (ImGui::Checkbox("Pathline", &this->pathlineRendering))
	{
		this->streamlineRendering = !this->pathlineRendering;

	}


	// Solver Options
	if (ImGui::InputText("File Path", solverOptions->filePath, sizeof(solverOptions->filePath)))
	{
	}

	if (ImGui::InputText("File Name", solverOptions->fileName, sizeof(solverOptions->fileName)))
	{
	}

	if (ImGui::InputInt3("Grid Size", solverOptions->gridSize, sizeof(solverOptions->gridSize)))
	{
	}
	if (ImGui::InputFloat3("Grid Diameter", solverOptions->gridDiameter, sizeof(solverOptions->gridDiameter)))
	{
		this->updateVolumeBox = true;
		this->updateRaycasting = true;
		this->updateStreamlines = true;
		this->updatePathlines = true;
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
		this->updatePathlines = true;
	}
	ImGui::SameLine();
	if (ImGui::InputInt("Last Index", &(solverOptions->lastIdx)))
	{
		this->updatePathlines = true;

	}
	if (ImGui::DragInt("Current Index", &(solverOptions->currentIdx), 1.0f, 0, solverOptions->lastIdx, "%d"))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
	}

	ImGui::PopItemWidth();

	if (ImGui::InputFloat("dt", &(solverOptions->dt)))
	{
		if (solverOptions->dt > 0)
		{
			this->updateStreamlines = true;
			this->updatePathlines = true;
		}
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
	if (this->streamlineRendering)
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


	if (ImGui::ListBox("Color Mode", &solverOptions->colorMode, ColorModeList, 4)){}


	// Show Lines
	if (this->streamlineRendering)
	{
		if (ImGui::Checkbox("Render Streamlines", &this->showStreamlines))
		{
			this->updateStreamlines = true;
		}
	}
	else // PathlineRendering
	{
		if (ImGui::Checkbox("Render Pathlines", &this->showPathlines))
		{
			this->updatePathlines = true;

		}
	}



	if (ImGui::Checkbox("Rendering Bounding box", &this->showRaycasting)){}


	if (ImGui::Button("Reset", ImVec2(80, 25)))
	{
		this->camera->SetPosition(0, 0, -10);
		this->camera->SetLookAtPos({ 0, 0, 0 });
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
	ImGui::SliderFloat("Tube Radius", &renderingOptions->tubeRadius, 0.0f, 0.1f);            // Edit 1 float using a slider from 0.0f to 1.0f



	ImGui::Text("Color Coding:");

	ImGui::ColorEdit4("Minimum", (float*)& renderingOptions->minColor);
	if (ImGui::InputFloat("Min Value", (float*)& renderingOptions->minMeasure, 0.1f)) {}

	ImGui::ColorEdit4("Maximum", (float*)& renderingOptions->maxColor);
	if (ImGui::InputFloat("Max Value", (float*)& renderingOptions->maxMeasure, 0.1f)) {}




	ImGui::End();

}

void RenderImGuiOptions::drawRaycastingOptions()
{


	ImGui::Begin("Raycasting Options");

	ImGui::DragFloat("Sampling Rate 0", &raycastingOptions->samplingRate_0,0.0001f);       
	ImGui::DragFloat("Isovalue 0", &raycastingOptions->isoValue_0, 0.01f);

	ImGui::Text("Surfaces color 0:");
	ImGui::ColorEdit4("Isosurface Color 0", (float*)& raycastingOptions->color_0);

	ImGui::End();

}