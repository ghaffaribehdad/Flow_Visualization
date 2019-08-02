#pragma once

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx11.h"
#include "..//SolverOptions.h"
#include <string>
#include "Graphics.h"


class RenderImGui
{
public:


	void drawSolverOptions(SolverOptions& solverOptions); // draw the solver option window
	void drawLog(Graphics* p_graphics);	// draw Log window
	void render(); // renders the imGui drawings
	// Log pointer
	char* log = new char[1000];
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
	if (p_graphics->fpsTimer.GetMilisecondsElapsed() > 1000.0)
	{
		fpsString = "FPS: " + std::to_string(fpsCounter);
		fpsCounter = 0;
		p_graphics->fpsTimer.Restart();
	}

	strcpy(this->log, fpsString.c_str());

	if (ImGui::InputTextMultiline("Frame per Second", this->log, 1000))
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