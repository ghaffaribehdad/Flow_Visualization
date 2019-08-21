#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "..//SolverOptions.h"
#include <string>
#include "Graphics.h"
#include "RenderingOptions.h"


class RenderImGui
{
public:


	void drawSolverOptions(SolverOptions& solverOptions); // draw the solver option window
	void drawLog(Graphics* p_graphics);	// draw Log window
	void drawLineRenderingOptions(RenderingOptions& renderingOptions, SolverOptions& solverOptions);
	void render(); // renders the imGui drawings

	// Log pointer
	char* log = new char[1000];

private:

};

void RenderImGui::drawSolverOptions(SolverOptions& solverOptions)
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Solver Options");

	ImGui::Text("Mode: ");
	ImGui::SameLine();

	//Solver Mode

	if (ImGui::Checkbox("Streamline", &solverOptions.streamline))
	{
		solverOptions.pathline = !solverOptions.streamline;
		solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		//p_graphics->InitializeScene();
	}
	ImGui::SameLine();
	if (ImGui::Checkbox("Pathline", &solverOptions.pathline))
	{
		solverOptions.streamline = !solverOptions.pathline;
		solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		//p_graphics->InitializeScene();
	}


	// Solver Options
	if (ImGui::InputText("File Path", solverOptions.filePath, sizeof(solverOptions.filePath)))
	{
	}

	if (ImGui::InputText("File Name", solverOptions.fileName, sizeof(solverOptions.fileName)))
	{
	}

	if (ImGui::InputInt3("Grid Size", solverOptions.gridSize, sizeof(solverOptions.gridSize)))
	{
	}
	if (ImGui::InputFloat3("Grid Diameter", solverOptions.gridDiameter, sizeof(solverOptions.gridDiameter)))
	{
	}

	if (ImGui::InputInt("precision", &(solverOptions.precision)))
	{

	}
	ImGui::PushItemWidth(75);
	if (ImGui::InputInt("First Index", &(solverOptions.firstIdx)))
	{

	}
	ImGui::SameLine();
	if (ImGui::InputInt("Last Index", &(solverOptions.lastIdx)))
	{

	}
	if (ImGui::DragInt("Current Index", &(solverOptions.currentIdx), 1.0f, 0, solverOptions.lastIdx, "%d"))
	{

	}
	ImGui::PopItemWidth();
	if (ImGui::InputFloat("dt", &(solverOptions.dt)))
	{
	}

	if (ImGui::InputInt("Lines", &(solverOptions.lines_count)))
	{

		if (solverOptions.pathline)
		{
			solverOptions.lineLength = solverOptions.lastIdx - solverOptions.firstIdx;
		}
		//p_graphics->InitializeScene();


	}
	if (solverOptions.streamline)
	{
		if (ImGui::InputInt("Line Length", &(solverOptions.lineLength)))
		{
			//p_graphics->InitializeScene();

		}
	}
	if (ImGui::ListBox("Color Mode", &solverOptions.colorMode, ColorModeList, 4))
	{

	}

	if (solverOptions.streamline)
	{
		if (ImGui::Checkbox("Render Streamlines", &solverOptions.beginStream))
		{
			solverOptions.userInterruption = true;
		}

	}
	else
	{
		if (ImGui::Checkbox("Render Pathlines", &solverOptions.beginPath))
		{
		}
	}

	if (ImGui::Checkbox("Rendering Bounding box", &solverOptions.beginRaycasting))
	{
		solverOptions.idChange = true;
	}


	if (ImGui::Button("Reset", ImVec2(80, 25)))
	{
		//this->p_graphics->camera.SetPosition(0, 0, -3);
	}


	ImGui::End();

}


void RenderImGui::drawLog(Graphics * p_graphics)
{
	ImGui::Begin("Log");



	// calculatin FPS
	static int fpsCounter = 0;
	static std::string fpsString = "FPS : 0";
	fpsCounter += 1;

	static float fps_array[10];
	static int fps_arrayCounter = 0;

	if (p_graphics->fpsTimer.GetMilisecondsElapsed() > 100.0)
	{
		fpsString = "FPS: " + std::to_string(10*fpsCounter);

		if (fps_arrayCounter < 10)
		{
			fps_array[fps_arrayCounter] = fpsCounter / 10;
			fps_arrayCounter++;
		}
		else
		{
			fps_arrayCounter = 0;
		}

		fpsCounter = 0;
		p_graphics->fpsTimer.Restart();
	}

	strcpy(this->log, fpsString.c_str());

	ImGui::PlotLines("Frame Times", fps_array, IM_ARRAYSIZE(fps_array),0,NULL,0,50,ImVec2(250,80));

	if (ImGui::InputTextMultiline("Frame per Second", this->log, 1000,ImVec2(100,35)))
	{

	}


	ImGui::End();
}


	

void RenderImGui::render()
{

	//Assemble Together Draw Data
	ImGui::Render();

	//Render Draw Data
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());


}


void RenderImGui::drawLineRenderingOptions(RenderingOptions& renderingOptions, SolverOptions& solverOptions)
{


	ImGui::Begin("Line Rendering Options");
	ImGui::SliderFloat("Tube Radius", &renderingOptions.tubeRadius, 0.0f, 0.4f);            // Edit 1 float using a slider from 0.0f to 1.0f



	ImGui::Text("Color Coding:");

	ImGui::ColorEdit4("Minimum", (float*)& renderingOptions.minColor);
	if (ImGui::InputFloat("Min Value", (float*)&renderingOptions.minMeasure, 0.1f)) {}

	ImGui::ColorEdit4("Maximum", (float*)& renderingOptions.maxColor);
	if (ImGui::InputFloat("Max Value", (float*)&renderingOptions.maxMeasure, 0.1f)) {}

	


	ImGui::End();

}