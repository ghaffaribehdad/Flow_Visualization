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

	if (ImGui::Checkbox("Streamline", &solverOptions->streamline))
	{
		solverOptions->pathline = !solverOptions->streamline;
		solverOptions->lineLength = solverOptions->lastIdx - solverOptions->firstIdx;
		//p_graphics->InitializeScene();
	}
	ImGui::SameLine();
	if (ImGui::Checkbox("Pathline", &solverOptions->pathline))
	{
		solverOptions->streamline = !solverOptions->pathline;
		solverOptions->lineLength = solverOptions->lastIdx - solverOptions->firstIdx;
		//p_graphics->InitializeScene();
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
		updateLineRendering = true;
	}

	if (ImGui::DragFloat3("Seed Box", solverOptions->seedBox, 0.01f))
	{
		updateLineRendering = true;
	}
	
	if (ImGui::DragFloat3("Seed Box Position", solverOptions->seedBoxPos, 0.01f))
	{
		updateLineRendering = true;
	}


	if (ImGui::InputInt("precision", &(solverOptions->precision)))
	{

	}
	ImGui::PushItemWidth(75);
	if (ImGui::InputInt("First Index", &(solverOptions->firstIdx)))
	{

	}
	ImGui::SameLine();
	if (ImGui::InputInt("Last Index", &(solverOptions->lastIdx)))
	{

	}
	if (ImGui::DragInt("Current Index", &(solverOptions->currentIdx), 1.0f, 0, solverOptions->lastIdx, "%d"))
	{
		updateLineRendering = true;
	}

	ImGui::PopItemWidth();

	if (ImGui::InputFloat("dt", &(solverOptions->dt)))
	{
		if (solverOptions->dt > 0)
		{
			updateLineRendering = true;
		}
	}

	if (ImGui::InputInt("Lines", &(solverOptions->lines_count)))
	{
		if (solverOptions->lines_count > 0)
		{
			updateLineRendering = true;

			if (solverOptions->pathline)
			{
				solverOptions->lineLength = solverOptions->lastIdx - solverOptions->firstIdx;
			}
		}



	}
	if (solverOptions->streamline)
	{
		if (ImGui::InputInt("Line Length", &(solverOptions->lineLength)))
		{

		}
	}
	if (ImGui::ListBox("Color Mode", &solverOptions->colorMode, ColorModeList, 4))
	{

	}

	if (solverOptions->streamline)
	{
		if (ImGui::Checkbox("Render Streamlines", &solverOptions->beginStream))
		{
			solverOptions->userInterruption = true;
		}

	}
	else
	{
		if (ImGui::Checkbox("Render Pathlines", &solverOptions->beginPath))
		{
		}
	}

	if (ImGui::Checkbox("Rendering Bounding box", &solverOptions->beginRaycasting))
	{
		solverOptions->idChange = true;
	}


	if (ImGui::Button("Reset", ImVec2(80, 25)))
	{
		this->camera->SetPosition(0, 0, -10);
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