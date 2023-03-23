#include "RenderImGuiOptions.h"
#include "..//Cuda/Cuda_helper_math_host.h"


void RenderImGuiOptions::drawSolverOptions()
{
	if (this->b_drawSolverOptions)
	{
		// Solver Options
		ImGui::Begin("Solver Options");

		if (ImGui::Combo("Select Field", &solverOptions->nField, ActiveField::ActiveFieldList, nFields))
		{

		}

		if (ImGui::Combo("Line Rendering Mode", &solverOptions->lineRenderingMode, LineRenderingMode::LineRenderingModeList, LineRenderingMode::LineRenderingMode::COUNT))
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

		if (solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::PATHLINES
			|| solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::STREAKLINES)
		{
			if (ImGui::Combo("Computation Mode", &solverOptions->computationMode, ComputationMode::ComputationModeList, ComputationMode::ComputationMode::COUNT))
			{

			}
		}



		if (ImGui::Checkbox("Fix Lines", &solverOptions->fixedLines))
		{
		}

		if (ImGui::InputText("Name", &solverOptions->outputFileName[0], 100 * sizeof(char)))
		{
		}

		if (ImGui::Button("Screenshot"))
		{
			this->saveScreenshot = true;
		}

		if (ImGui::InputInt("Range", &this->screenshotRange, 1, 10))
		{
		}

		if (ImGui::Combo("Projection", &solverOptions->projection, Projection::ProjectionList, Projection::Projection::COUNT))
		{
			this->updateOIT = true;
		}

		solverOptions->streakBoxPos[0] = 0;
		if (solverOptions->projection == Projection::Projection::STREAK_PROJECTION)
		{

			float init_pos = -1 * (solverOptions->gridDiameter[0] / solverOptions->gridSize[0]) * (solverOptions->projectPos - solverOptions->gridSize[0] / 2.0f);
			init_pos -= solverOptions->timeDim / 2;
			init_pos += (solverOptions->currentIdx - solverOptions->firstIdx) * (solverOptions->timeDim / (solverOptions->lastIdx - solverOptions->firstIdx));
			solverOptions->streakBoxPos[0] = init_pos;
			solverOptions->streakBox[0] = 0;

		}
		else if (solverOptions->projection == Projection::Projection::STREAK_PROJECTION_FIX)
		{
			solverOptions->streakBox[0] = 0;
		}
		else
		{
			solverOptions->streakBox[0] = solverOptions->gridDiameter[0];
		}

		if (raycastingOptions->raycastingMode == RaycastingMode::PROJECTION_FORWARD		||
			raycastingOptions->raycastingMode == RaycastingMode::PROJECTION_BACKWARD	||
			raycastingOptions->raycastingMode == RaycastingMode::PROJECTION_AVERAGE		||
			raycastingOptions->raycastingMode == RaycastingMode::PROJECTION_LENGTH
			)
		{
			solverOptions->streakBoxPos[0] = (raycastingOptions->projectionPlanePos - solverOptions->gridSize[0]/2) * (solverOptions->gridDiameter[0] / (solverOptions->gridSize[0]));
		}

		if (ImGui::InputFloat("Time Dim", &solverOptions->timeDim))
		{

		}

		if (ImGui::InputInt("Projection Plane", &solverOptions->projectPos, 1, 5))
		{
			//this->updateStreamlines = true;
			//this->updatePathlines = true;
			//this->updateStreaklines = true;
		}



		if (ImGui::Checkbox("Periodic", &solverOptions->periodic))
		{

		}



		if (ImGui::Checkbox("pause updating", &solverOptions->updatePause))
		{

		}




		if (ImGui::DragFloat3("Velocity Scaling Factor", solverOptions->velocityScalingFactor, 0.01f))
		{
			this->updateVolumeBox = true;
			this->updateRaycasting = true;
			this->updateStreamlines = true;
			this->updatePathlines = true;
			this->updateStreaklines = true;

		}
		if (ImGui::Button("Optical Flow"))
		{
			solverOptions->velocityScalingFactor[0] = (1.0f / solverOptions->gridSize[0]) * solverOptions->gridDiameter[0]; // 20 pixel padding
			solverOptions->velocityScalingFactor[1] = (1.0f / solverOptions->gridSize[1]) * solverOptions->gridDiameter[1]; // 20 pixel padding
			solverOptions->velocityScalingFactor[2] = (1.0f / solverOptions->gridSize[2]) * solverOptions->gridDiameter[2]; // 20 pixel padding

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
		if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_RANDOM)
		{
			if (ImGui::Checkbox("Random RNG", &solverOptions->randomSeed))
			{
				this->updateStreamlines = true;
				this->updatePathlines = true;
				this->updateStreaklines = true;
			}
		}

		if (!solverOptions->randomSeed)
		{
			ImGui::InputInt("Seed Value", &solverOptions->seedValue, 1, 10);
		}

		if (solverOptions->seedingPattern == (int)SeedingPattern::SEED_GRIDPOINTS)
		{
			if (ImGui::DragInt3("Seed Grid", solverOptions->seedGrid, 1, 1, 1024))
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


		if (ImGui::DragInt("First Index", &(solverOptions->firstIdx),1,solverOptions->currentIdx,solverOptions->lastIdx-1))
		{


		}

		if (solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::STREAKLINES || solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::PATHLINES)
		{
			solverOptions->lineLength = solverOptions->lastIdx - solverOptions->firstIdx;
		}

		ImGui::SameLine();

		if (ImGui::DragInt("Last Index", &(solverOptions->lastIdx),1,solverOptions->firstIdx-1))
		{
			this->updatePathlines = true;
			this->updateStreaklines = true;

		}


		if (solverOptions->isEnsemble)
		{
			if (ImGui::DragInt("First Member Index", &solverOptions->firstMemberIdx))
			{


			}

			if (ImGui::DragInt("Last Member Index", &solverOptions->lastMemberIdx))
			{

			}
		}
		if (ImGui::InputInt("Current Index", &(solverOptions->currentIdx)))
		{
			if (solverOptions->currentIdx > solverOptions->lastIdx)
			{
				solverOptions->currentIdx = solverOptions->lastIdx;
			}

			if (solverOptions->currentIdx < solverOptions->firstIdx)
			{
				solverOptions->currentIdx = solverOptions->firstIdx;
			}
			this->updateStreamlines = true;
			solverOptions->loadNewfile = true;
			this->solverOptions->fileChanged = true;
			this->updateFile[0] = true;


			this->updateTimeSpaceField = true;
			this->updateRaycasting = true;
			this->crossSectionOptions->updateTime = true;
			this->updatefluctuation = true;
			this->updateCrossSection = true;
			this->updateOIT = true;
			this->saved = false;
			this->spaceTimeOptions->volumeLoaded = false;
			this->solverOptions->fileChanged = true;
			this->updateVisitationMap = true;

		}


		if (solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::STREAMLINES)
		{
			if (ImGui::InputInt("Current segment", &solverOptions->currentSegment, 1, 10))
			{
				if (solverOptions->currentSegment < 0)
					solverOptions->currentSegment = 1;
				else if (solverOptions->currentSegment > solverOptions->lineLength)
					solverOptions->currentSegment = solverOptions->lineLength;
			}

		}


		ImGui::PopItemWidth();

		if (ImGui::DragFloat("dt", &(solverOptions->dt),0.0001f, 0.000001f, 1.0f, "%.6f"))
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

		if (solverOptions->lineRenderingMode == LineRenderingMode::LineRenderingMode::STREAMLINES)
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


		if (ImGui::Checkbox("Using Transparency", &solverOptions->usingTransparency))
		{
			this->updateOIT = true;
		}

		if (ImGui::Combo("Transparency Mode", &solverOptions->transparencyMode, TransparencyMode::TransparencyModeList, TransparencyMode::TransparencyMode::COUNT))
		{

		}

		if (ImGui::Checkbox("Using Threshold", &solverOptions->usingThreshold))
		{

		}
		if (solverOptions->usingThreshold)
		{
			if (ImGui::InputFloat("Threshold", &solverOptions->transparencyThreshold, 0.01f, 0.1f))
			{

			}
		}

		if (ImGui::Combo("Color\\Transparency Mode", &solverOptions->colorMode, ColorMode::ColorModeList, ColorMode::ColorMode::COUNT))
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
		case LineRenderingMode::LineRenderingMode::STREAMLINES:
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

		case LineRenderingMode::LineRenderingMode::PATHLINES:
		{
			if (this->solverOptions->lastIdx - this->solverOptions->firstIdx >= 1)
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


		case LineRenderingMode::LineRenderingMode::STREAKLINES:
		{
			if (this->solverOptions->lastIdx - this->solverOptions->firstIdx >= 1)
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
			this->camera->SetPosition(-3.91f, 0.05f, -4.94f);
			this->camera->SetLookAtPos({ 0.54f, -0.02f, 0.84f });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;
		}

		ImGui::SameLine();

		if (ImGui::Button("Edge View", ImVec2(80, 25)))
		{
			this->camera->SetPosition(-5.18f, 0.71f, -7.42f);
			this->camera->SetLookAtPos({ 0.75f,-0.35f,0.55f });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;

		}
		ImGui::SameLine();

		if (ImGui::Button("Top", ImVec2(80, 25)))
		{
			this->camera->SetPosition(0, 12, 0);
			this->camera->SetLookAtPos({ 0, -1.0, 0 });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;

		}

		ImGui::SameLine();
		if (ImGui::Button("Bottom", ImVec2(80, 25)))
		{
			this->camera->SetPosition(0, 0, -10);
			this->camera->SetLookAtPos({ 0, 0, 0 });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;

		}

		if (ImGui::Button("Side", ImVec2(80, 25)))
		{
			this->camera->SetPosition(6.7f, 3.40f, 9.3f);
			this->camera->SetLookAtPos({ -0.56f, -.37f, -0.74f });

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
		ImGui::SameLine();
		if (ImGui::Button("Top Zoom", ImVec2(80, 25)))
		{
			this->camera->SetPosition(0, 1, 0);
			this->camera->SetLookAtPos({ 0, -1.0, 0 });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;

		}
		ImGui::SameLine();
		if (ImGui::Button("Edge View Positive", ImVec2(80, 25)))
		{
			this->camera->SetPosition(-5.18f, 5.71f, -8.42f);
			this->camera->SetLookAtPos({ 0.75f,-0.35f,0.55f });

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updatefluctuation = true;
			this->updateFTLE = true;

		}


		ImGui::End();
	}
}


void RenderImGuiOptions::drawFieldOptions()
{
	ImGui::Begin("Field Options");


	if (ImGui::Button("Set Default", ImVec2(100, 30)))
	{
		setArray<float>(&this->solverOptions->seedBox[0], &fieldOptions[0].gridDiameter[0]);
		setArray<float>(&this->raycastingOptions->clipBox[0], &fieldOptions[0].gridDiameter[0]);

		this->solverOptions->gridDiameter[0] = fieldOptions[0].gridDiameter[0];
		this->solverOptions->gridDiameter[1] = fieldOptions[0].gridDiameter[1];
		this->solverOptions->gridDiameter[2] = fieldOptions[0].gridDiameter[2];
		this->solverOptions->gridSize[0] = fieldOptions[0].gridSize[0];
		this->solverOptions->gridSize[1] = fieldOptions[0].gridSize[1];
		this->solverOptions->gridSize[2] = fieldOptions[0].gridSize[2];
		this->solverOptions->firstIdx = fieldOptions[0].firstIdx;
		this->solverOptions->lastIdx = fieldOptions[0].lastIdx;
	}

	ImGui::SameLine();

	if (ImGui::Button("Add Fields", ImVec2(100, 30)))
	{
		if (nFields < 4)
		{
			this->nFields++;
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Remove Fields", ImVec2(100, 30)))
	{
		if (nFields > 1)
		{
			this->nFields--;
		}
	}

	for (int i = 0; i < nFields; i++)
	{

			if (ImGui::Combo((std::string("Dataset") + std::to_string(i)).c_str(), reinterpret_cast<int*>(&this->dataset[i]), Dataset::datasetList, Dataset::Dataset::COUNT))
			{
				updateFile[i] = true;

				switch (dataset[i])
				{

				case Dataset::Dataset::KIT2REF_COMP:
					this->fieldOptions[i].setField("Comp_FieldP", "G:\\Dataset_Compressed\\KIT2\\Comp_Ref\\", 192, 192, 192, 7.854f, 2.0f, 3.1415f, 1, 1000, 0.001f, true, 7000000);
					break;
				case Dataset::Dataset::KIT2OW_COMP:
					this->fieldOptions[i].setField("Comp_FieldP", "G:\\Dataset_Compressed\\KIT2\\Comp_OW\\", 192, 192, 192, 7.854f, 2.0f, 3.1415f, 1, 1000, 0.001f, true, 7000000);
					break;
				case Dataset::Dataset::KIT2BF_COMP:
					this->fieldOptions[i].setField("Comp_FieldP", "G:\\Dataset_Compressed\\KIT2\\Comp_Ref\\", 192, 192, 192, 7.854f, 2.0f, 3.1415f, 1, 1000, 0.001f, true, 7000000);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_AVG_262:
					this->fieldOptions[i].setField("OF_temperature_dt_avg_262_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_AVG_263:
					this->fieldOptions[i].setField("OF_temperature_dt_avg_263_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_AVG_264:
					this->fieldOptions[i].setField("OF_temperature_dt_avg_264_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_VIS_262:
					this->fieldOptions[i].setField("OF_temperature_dt_vis_262_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_VIS_263:
					this->fieldOptions[i].setField("OF_temperature_dt_vis_263_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_OF_VIS_264:
					this->fieldOptions[i].setField("OF_temperature_dt_vis_264_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
					break;
				case Dataset::Dataset::GRAND_ENSEMBLE_TIME:
					this->fieldOptions[i].setFieldEnsemble("member_", "F:\\Grand_ensemble\\", "time_", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 1, 20, 0.001f);
					break;
				case Dataset::Dataset::MUTUAL_INFO:
					this->fieldOptions[i].setField("mi_", "E:\\MutualInfoField\\", 250, 20, 352, 3.0f, 2.0f, 3.0f,0, 1, 0.001f);
					break;
				case Dataset::Dataset::MUTUAL_INFO_1:
					this->fieldOptions[i].setField("mi_", "E:\\MutualInfoField\\", 250, 20, 352, 3.0f, 2.0f, 3.0f, 1, 1, 0.001f);
					break;
				case Dataset::Dataset::KIT3_COMPRESSED:
					this->fieldOptions[i].setField("Fluc_Comp_", "G:\\Dataset_Compressed\\KIT3\\Comp_Fluc\\", 64, 503, 2048, 0.4f, 2.0f, 7.0f, 500, 1000, 0.001f, true, 64000000);
					break;
				case Dataset::Dataset::KIT3_RAW:
					this->fieldOptions[i].setField("FieldP", "G:\\KIT3\\Initial\\", 64, 503, 2048, 0.4f, 2.0f, 7.0f, 500, 1000, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG:
					this->fieldOptions[i].setField("Field_AVG", "E:\\TUI_RBC_Small\\timeAVG\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_50:
					this->fieldOptions[i].setField("FieldAVG_50_", "Y:\\RBC\\AVG50\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_20:
					this->fieldOptions[i].setField("FieldAVG_20_", "Y:\\RBC\\AVG20\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_100:
					this->fieldOptions[i].setField("FieldAVG_100_", "Y:\\RBC\\AVG100\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_OF_20:
					this->fieldOptions[i].setField("AVG20_OF_", "Y:\\RBC\\AVG20_OF\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_OF_50:
					this->fieldOptions[i].setField("AVG50_OF_", "Y:\\RBC\\AVG50_OF\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_OF_100:
					this->fieldOptions[i].setField("AVG100_OF_", "Y:\\RBC\\AVG100_OF\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_OF:
					this->fieldOptions[i].setField("OF_temperature", "E:\\TUI_RBC_Small\\OF\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG_OF_600:
					this->fieldOptions[i].setField("Field_OF_AVG600_", "E:\\TUI_RBC_Small\\timeAVG\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::RBC_AVG500:
					this->fieldOptions[i].setField("Field_OF_AVG600_", "E:\\TUI_RBC_Small\\timeAVG\\", 1024, 32, 1024, 5.0f, 0.2f, 5.0f, 1, 50, 0.001f);
					break;
				case Dataset::Dataset::SMOKE_00050_ENSEMBLE:
					this->fieldOptions[i].setField("denisty50_padded_", "G:\\Smoke_ensembleData\\density_bin_000050\\", 100, 178, 100, 2.0f, 3.56f, 2.0f, 1, 31, 0.001f);
					break;
				case Dataset::Dataset::KIT3_CC:
					this->fieldOptions[i].setField("FieldP_CC_Decomp_", "E:\\Compression_Report\\SingleKIT3\\ParticleTracing\\", 64, 503, 2048, 0.4f, 2.0f, 7.0f, 800, 800, 0.001f);
					break;
				case Dataset::Dataset::KIT3_CUSZ:
					this->fieldOptions[i].setField("FieldP_CCSZ_Decomp_", "E:\\Compression_Report\\SingleKIT3\\ParticleTracing\\", 64, 503, 2048, 0.4f, 2.0f, 7.0f, 800, 800, 0.001f);
					break;
				}

				if (i == 0)
				{
					setArray<float>(&this->solverOptions->seedBox[0], &fieldOptions[0].gridDiameter[0]);
					setArray<float>(&this->raycastingOptions->clipBox[0], &fieldOptions[0].gridDiameter[0]);

					this->solverOptions->gridDiameter[0] = fieldOptions[0].gridDiameter[0];
					this->solverOptions->gridDiameter[1] = fieldOptions[0].gridDiameter[1];
					this->solverOptions->gridDiameter[2] = fieldOptions[0].gridDiameter[2];
					this->solverOptions->gridSize[0] = fieldOptions[0].gridSize[0];
					this->solverOptions->gridSize[1] = fieldOptions[0].gridSize[1];
					this->solverOptions->gridSize[2] = fieldOptions[0].gridSize[2];
					this->solverOptions->firstIdx = fieldOptions[0].firstIdx;
					this->solverOptions->lastIdx = fieldOptions[0].lastIdx;
					this->solverOptions->currentIdx = fieldOptions[0].firstIdx;
				}
			}
		
		if (ImGui::DragInt((std::string("Current index ") + std::to_string(i)).c_str(), &fieldOptions[i].currentIdx,1, fieldOptions[i].firstIdx, fieldOptions[i].lastIdx))
		{
			updateFile[i] = true;
		}

		if (ImGui::Checkbox((std::string("Is Compressed ") + std::to_string(i)).c_str(), &fieldOptions[i].isCompressed))
		{
		}

		if (ImGui::Checkbox((std::string("Is Ensemble ") + std::to_string(i)).c_str(), &fieldOptions[i].isEnsemble))
		{
		}

		if (ImGui::InputText((std::string("File Path ") + std::to_string(i)).c_str(), &fieldOptions[i].filePath[0], 100 * sizeof(char)))
		{
		}

		if (ImGui::InputText((std::string("File Name ") + std::to_string(i)).c_str(), &fieldOptions[i].fileName[0], 100 * sizeof(char)))
		{
		}

		if (solverOptions->isEnsemble)
		{
			if (ImGui::InputText((std::string("Subpath ") + std::to_string(i)).c_str(), &fieldOptions[i].subpath[0], 100 * sizeof(char)))
			{
			}
		}

		if (ImGui::InputInt3((std::string("Grid Size ") + std::to_string(i)).c_str(), fieldOptions[i].gridSize, sizeof(&fieldOptions[i].gridSize)))
		{
			this->updateSeedBox = true;
			this->updateStreamlines = true;
			this->updatePathlines = true;
			this->updateStreaklines = true;
		}
		if (ImGui::InputFloat3((std::string("Grid Diameter ") + std::to_string(i)).c_str(), fieldOptions[i].gridDiameter, sizeof(&fieldOptions[i].gridDiameter)))
		{
			this->updateVolumeBox = true;
			this->updateRaycasting = true;
			this->updateStreamlines = true;
			this->updatePathlines = true;
		}
		ImGui::NewLine();
	}



	ImGui::End();
}

void RenderImGuiOptions::drawPathSpaceTime()
{
	ImGui::Begin("Path-Surface Options");

	if (pathSpaceTimeOptions->lastIdx - pathSpaceTimeOptions->firstIdx > 2)
	{
		if (ImGui::Checkbox("Enable Path Space-Time", &this->showPathSpaceTime))
		{
			this->updatePathSpaceTime = true;
		}
	}


	if (ImGui::Checkbox("Color-coding path surface", &pathSpaceTimeOptions->colorCoding))
	{
		this->updatePathSpaceTime = true;
	}
	if (ImGui::DragInt("time position", &pathSpaceTimeOptions->timeStep,1,0, pathSpaceTimeOptions->timeGrid - 1))
	{
		this->updatePathSpaceTime = true;
	}

	if (ImGui::DragFloat("sigma", &pathSpaceTimeOptions->sigma, 0.01, 0.0f, 10.0f, "%3f"))
	{
		this->updatePathSpaceTime = true;
	}
	ImGui::End();
}
void RenderImGuiOptions::drawLog()
{

	if (b_drawLog)
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
	if (b_drawLineRenderingOptions)
	{
		ImGui::Begin("Rendering Options");

		if (ImGui::Button("Remote connection"))
		{
			renderingOptions->mouseSpeed = 0.0005f;
		}
		if (ImGui::SliderFloat("Mouse Speed", &renderingOptions->mouseSpeed, 0.00005f, 0.05f, "%.5f"))
		{
		}

		if (ImGui::Checkbox("Show Seed Box", &renderingOptions->showSeedBox))
		{
		}

		if (ImGui::Checkbox("Show Streak Box", &renderingOptions->showStreakBox))
		{
		}

		if (ImGui::Checkbox("Show Volume Box", &renderingOptions->showVolumeBox))
		{
		}

		if (ImGui::Checkbox("Show Clip Box", &renderingOptions->showClipBox))
		{
		}

		if (ImGui::Checkbox("Show Streak plane", &renderingOptions->showStreakPlane))
		{
		}


		if (ImGui::SliderFloat("Box Radius", &renderingOptions->boxRadius, 0.0f, 0.05f, "%.4f"))
		{
		}

		if (ImGui::SliderFloat("FOV", &renderingOptions->FOV_deg,0,180.0, "%.4f"))
		{
			updateRaycasting = true;
		}		
		
		if (ImGui::SliderFloat("Far Field", &renderingOptions->farField,0.0f,100000.0f, "%.4f"))
		{
			updateRaycasting = true;
		}

		if (ImGui::SliderFloat("Near Field", &renderingOptions->nearField, 0.000001f, 1.0f, "%.4f"))
		{
			updateRaycasting = true;
		}



		if (ImGui::Combo("Rendering Mode", &renderingOptions->renderingMode, RenderingMode::RenderingModeList, RenderingMode::RenderingMode::COUNT))
		{
			this->updateShaders = true;
		}

		if (ImGui::Combo("Draw Mode", &renderingOptions->drawMode, DrawMode::DrawModeList, DrawMode::DrawMode::COUNT))
		{
		}


		if (ImGui::InputInt("Line Length", &renderingOptions->lineLength, 1, 10))
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


		if (ImGui::SliderFloat("Tube Radius", &renderingOptions->tubeRadius, 0.0f, 0.1f, "%.5f"))
		{
			this->updateOIT = true;
		}



		ImGui::Text("Color Coding:");

		if (ImGui::ColorEdit4("Light Color", (float*)&renderingOptions->lightColor))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		if (ImGui::ColorEdit4("Minimum", (float*)&renderingOptions->minColor))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;

		}
		if (ImGui::InputFloat("Min Value", (float*)& renderingOptions->minMeasure, 0.1f))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;

		}

		if (ImGui::ColorEdit4("Maximum", (float*)&renderingOptions->maxColor))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;

		}
		if (ImGui::InputFloat("Max Value", (float*)& renderingOptions->maxMeasure, 0.1f))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;

		}


		if (ImGui::SliderFloat("Specular0", &renderingOptions->Ks, 0.0f, 2, "%.2f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		if (ImGui::SliderFloat("Specular1", &renderingOptions->Ks1, 0.0f, 2, "%.2f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		if (ImGui::SliderFloat("Diffuse0", &renderingOptions->Kd, 0.0f, 1, "%.2f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;

		}
		if (ImGui::SliderFloat("Diffuse1", &renderingOptions->Kd1, 0.0f, 1, "%.2f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		if (ImGui::SliderFloat("Ambient", &renderingOptions->Ka, 0.0f,1, "%.2f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		if (ImGui::SliderFloat("Shininess", &renderingOptions->shininess, 1.0f, 10, "%.1f"))
		{
			this->updateOIT = true;
			this->updateRaycasting = true;
		}

		ImGui::End();
	}

}

void RenderImGuiOptions::drawRaycastingOptions()
{

	if (b_drawRaycastingOptions)
	{
		ImGui::Begin("Raycasting Options");

		if (ImGui::Checkbox("Enable Raycasintg", &this->showRaycasting))
		{
			this->updateRaycasting = true;
			if (!this->showRaycasting)
			{
				this->releaseRaycasting = true;
			}
		}


		if (ImGui::Combo("Isosurface field 0", &raycastingOptions->raycastingField_0, ActiveField::ActiveFieldList, nFields))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updateFile[0];
		}

		if (raycastingOptions->raycastingMode == RaycastingMode::DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::PLANAR_DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE_TEMP ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_SEPARATE ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_ADVANCED ||
			raycastingOptions->raycastingMode == RaycastingMode::DVR_DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_TRANSPARENCY ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE_DEFECT

			)
		{
			if (ImGui::Combo("Isosurface field 1", &raycastingOptions->raycastingField_1, ActiveField::ActiveFieldList, nFields))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
			}
		}


		if (ImGui::Combo("Mode", &raycastingOptions->raycastingMode, RaycastingMode::modeList, RaycastingMode::Mode::COUNT))
		{
			this->updateRaycasting = true;

			if (raycastingOptions->raycastingMode == RaycastingMode::Mode::PLANAR)
			{
				raycastingOptions->samplingRate_0 = 0.0001f;
			}
			else
			{
				raycastingOptions->samplingRate_0 = 0.001f;
			}

		}


		if (ImGui::SliderFloat("Brightness", &raycastingOptions->brightness, 0.2f, 2))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;

		}

		if (ImGui::SliderFloat("Projection Plane Pos", &raycastingOptions->projectionPlanePos,0.0f, (float)solverOptions->gridSize[0]))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;

		}


		if (ImGui::SliderFloat("Reflection Coefficient", &raycastingOptions->reflectionCoefficient, 0, 2.0f))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;

		}
		if (ImGui::Checkbox("Inside only", &this->raycastingOptions->insideOnly))
		{
			this->updateRaycasting = true;
		}

		if (ImGui::Checkbox("Enable Adaptive Sampling", &this->raycastingOptions->adaptiveSampling))
		{
			this->updateRaycasting = true;
		}

		if (ImGui::Checkbox("within", &this->raycastingOptions->within))
		{
			this->updateRaycasting = true;
		}

		if (ImGui::Checkbox("diff importance", &this->raycastingOptions->diffImportance))
		{
			this->updateRaycasting = true;


		}
		if (raycastingOptions->diffImportance)
		{
			if (ImGui::SliderFloat("diff limit", &raycastingOptions->limit_difference, 0,20))
			{
				this->updateRaycasting = true;
			}
		}


		if (ImGui::Checkbox("Enable Normal Curves", &this->raycastingOptions->normalCurves))
		{
			this->updateRaycasting = true;
		}
		if (raycastingOptions->normalCurves)
		{
			if (ImGui::DragFloat("max distance", &raycastingOptions->max_distance,0.001f, 0, 1))
			{
				this->updateRaycasting = true;
			}
		}


		if (ImGui::Checkbox("Secondary Only", &this->raycastingOptions->secondaryOnly))
		{
			this->updateRaycasting = true;
		}

		//if (this->raycastingOptions->normalCurves)
		//{
		//	if (ImGui::SliderFloat("Distance to Primary", &raycastingOptions->distanceToPrimary, 0, 0.1f,"%6f"))
		//	{
		//		this->updateRaycasting = true;
		//		this->updateTimeSpaceField = true;

		//	}
		//}

		if (ImGui::Combo("Isosurface Measure 0", &raycastingOptions->isoMeasure_0, IsoMeasure::IsoMeasureModes, (int)IsoMeasure::COUNT))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;

		}


		if (raycastingOptions->raycastingMode == RaycastingMode::DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE ||
			raycastingOptions->raycastingMode == RaycastingMode::DVR_DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::PLANAR_DOUBLE ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE_TEMP ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_SEPARATE ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_ADVANCED ||
			raycastingOptions->raycastingMode == RaycastingMode::DOUBLE_TRANSPARENCY ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE_DEFECT

			)
		{
			if (ImGui::Combo("Isosurface Measure 1", &raycastingOptions->isoMeasure_1, IsoMeasure::IsoMeasureModes, (int)IsoMeasure::COUNT))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
			}

		}



		if (raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE ||
			raycastingOptions->raycastingMode == RaycastingMode::MULTISCALE_DEFECT)
		{
			if (ImGui::Combo("Field Level Isosurface0", &raycastingOptions->fieldLevel_0, IsoMeasure::FieldLevelList, (int)IsoMeasure::COUNT_LEVEL))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
			}	
			if (ImGui::Combo("Field Level Isosurface1", &raycastingOptions->fieldLevel_1, IsoMeasure::FieldLevelList, (int)IsoMeasure::COUNT_LEVEL))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
			}
			if (ImGui::Combo("Field Level Isosurface2", &raycastingOptions->fieldLevel_2, IsoMeasure::FieldLevelList, (int)IsoMeasure::COUNT_LEVEL))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
			}


		}

		if (ImGui::SliderFloat("Transparency 0", &raycastingOptions->transparency_0,0,1.0f))
		{
			this->updateRaycasting = true;
		}
		if (ImGui::DragFloat("Amplitude 0", &raycastingOptions->amplitude,0.001f,0.0001f,10.0f))
		{
			this->updateRaycasting = true;
		}
		if (ImGui::DragFloat("Relax 0", &raycastingOptions->relax, 0.001f, 0.0001f, 5.0f))
		{
			this->updateRaycasting = true;
		}
		if (ImGui::SliderFloat("Transparency 1", &raycastingOptions->transparency_1, 0, 1.0f))
		{
			this->updateRaycasting = true;
		}


		if (ImGui::DragFloat("Sampling Rate Projection", &raycastingOptions->samplingRate_projection, 0.001f, 0.001f, 1.0f, "%.3f"))
		{
			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updateFTLE = true;
		}

		if (ImGui::DragFloat("Sampling Rate 0", &raycastingOptions->samplingRate_0, 0.00001f, 0.0001f, 1.0f, "%.5f"))
		{
			if (raycastingOptions->samplingRate_0 < 0.0001f)
			{
				raycastingOptions->samplingRate_0 = 0.0001f;
			}
			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updateFTLE = true;



		}

		if (ImGui::DragFloat("Sampling Rate 1", &raycastingOptions->samplingRate_1, 0.00001f, 0.0001f, 1.0f, "%.5f"))
		{

			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updateFTLE = true;
		}

		if (ImGui::Checkbox("binary search", &raycastingOptions->binarySearch))
		{
			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updateFTLE = true;
		}

		if (ImGui::DragInt("max iteration", &raycastingOptions->maxIteration,10,1,100))
		{
			this->updateRaycasting = true;
			this->updateDispersion = true;
			this->updateFTLE = true;
		}

		if (ImGui::DragFloat3("Clip Box", raycastingOptions->clipBox, 0.01f))
		{

			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatefluctuation = true;

		}

		if (ImGui::DragFloat3("Clip Box Center", raycastingOptions->clipBoxCenter, 0.01f))
		{

			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatefluctuation = true;

		}


		if (ImGui::DragFloat("Isovalue 0", &raycastingOptions->isoValue_0, 0.001f))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatePathSpaceTime = true;
		}

		if (ImGui::DragFloat("Isovalue 1", &raycastingOptions->isoValue_1, 0.001f))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
		}


		if (ImGui::DragFloat("Tolerance 0", &raycastingOptions->tolerance_0, 0.000001f, 0.0001f, 5, "%5f"))
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

		if (ImGui::ColorEdit3("Isosurface Color 1", (float*)& raycastingOptions->color_1))
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



		if (this->solverOptions->fileLoaded)
		{
			ImGui::Text("File is loaded!");
		}
		else
		{
			ImGui::Text("File is not loaded yet!");
		}



		if (ImGui::Checkbox("Color Coding Range", &this->raycastingOptions->ColorCodingRange))
		{
			this->updateRaycasting = true;
		}


		if (ImGui::Combo("Projection Plane", &raycastingOptions->projectionPlane, IsoMeasure::ProjectionPlaneList, IsoMeasure::COUNT_PLANE))
		{
			this->updateRaycasting = true;
		}

		if (ImGui::InputFloat("Min Value", (float*)&raycastingOptions->minVal, 0.001f, 0.1f))
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

		if (ImGui::InputFloat("Planar Probe Position", &raycastingOptions->planeProbePosition, 0.01f, 0.1f))
		{
			if (raycastingOptions->planeProbePosition > 1.0f)
			{
				raycastingOptions->planeProbePosition = 1.0f;
			}
			else if (raycastingOptions->planeProbePosition < 0.0f)
			{
				raycastingOptions->planeProbePosition = 0.0f;
			}

			this->updateRaycasting = true;
		}
		



		ImGui::End();
	}
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
			
			this->updateDispersion = true;
			this->dispersionOptions->released = false;
		}
	}


	if (solverOptions->lastIdx - solverOptions->firstIdx > 0)
	{
		if (ImGui::Checkbox("Enable FTLE rendering", &this->showFTLE))
		{
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
	if (b_drawTimeSpaceOptions)
	{


		ImGui::Begin("Time-Space Rendering");

		if (ImGui::DragFloat("Light color", (float*)& spaceTimeOptions->brightness,0.01f,1,3))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatefluctuation = true;
		}

		if (ImGui::Checkbox("Gaussin Filtering", &spaceTimeOptions->gaussianFilter))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatefluctuation = true;
		}

		if (spaceTimeOptions->gaussianFilter)
		{
			if (ImGui::DragInt("filter size", &spaceTimeOptions->filterSize, 1, 1, 50))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
				this->updatefluctuation = true;
			}

			if (ImGui::DragFloat("Standard Deviation ", &spaceTimeOptions->std,0.5f,0.5f,50.0f))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
				this->updatefluctuation = true;
			}
		}

		if (ImGui::Checkbox("Gaussin Filtering Height", &spaceTimeOptions->gaussianFilterHeight))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
			this->updatefluctuation = true;
		}

		if (spaceTimeOptions->gaussianFilterHeight)
		{
			if (ImGui::DragInt("filter size Height", &spaceTimeOptions->filterSizeHeight, 1, 1, 50))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
				this->updatefluctuation = true;
			}

			if (ImGui::DragFloat("Standard Deviation Height", &spaceTimeOptions->stdHeight, 0.5f, 0.5f, 50.0f))
			{
				this->updateRaycasting = true;
				this->updateTimeSpaceField = true;
				this->updatefluctuation = true;
			}
		}



		if (solverOptions->lastIdx - solverOptions->firstIdx > 0)
		{
			if (ImGui::Checkbox("Enable Rendering", &this->showFluctuationHeightfield))
			{
				this->updatefluctuation = true;
			}
			
		}

		if (ImGui::Checkbox("Shift projection plane", &spaceTimeOptions->shiftProjection))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::Checkbox("Shift space-time", &spaceTimeOptions->shifSpaceTime))
		{
			this->updatefluctuation = true;
		}
			   
		if (ImGui::Checkbox("Render Isosurfaces", &spaceTimeOptions->additionalRaycasting))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::Combo("Height Mode", &spaceTimeOptions->heightMode, IsoMeasure::IsoMeasureModes, IsoMeasure::COUNT))
		{
			this->updatefluctuation=true;
		}
		if (ImGui::Checkbox("Shading", &spaceTimeOptions->shading))
		{
			this->updatefluctuation = true;
		}


		if (ImGui::Combo("Slider Background", &spaceTimeOptions->sliderBackground, SliderBackground::SliderBackgroundList, SliderBackground::SliderBackground::COUNT))
		{
			this->updatefluctuation = true;
		}

		if (spaceTimeOptions->sliderBackground == SliderBackground::SliderBackground::BAND)
		{
			if (ImGui::InputInt("Band Layers", &spaceTimeOptions->bandSize,1,2))
			{
				this->updatefluctuation = true;
			}

		}

		if(ImGui::InputFloat("Time Dimenstion",&solverOptions->timeDim,0.1f,0.2f))


		if (ImGui::InputInt("Number of Slices", &spaceTimeOptions->streamwiseSlice, 1, 1))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::DragFloat("Slice Position", &spaceTimeOptions->streamwiseSlicePos, 0.1f, 0))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::DragFloat("Height Tolerance", &spaceTimeOptions->hegiht_tolerance, 0.0001f, 0.0001f, 1, "%8f"))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::DragFloat("sampling ration time", &spaceTimeOptions->samplingRatio_t, 0.001f, 0.001f, 5, "%4f"))
		{
			this->updatefluctuation = true;
		}

		if (ImGui::DragFloat("isovalue", &spaceTimeOptions->isoValue, 0.001f, 0.001f, 5, "%4f"))
		{
			this->updatefluctuation = true;
		}

		ImGui::Text("Color Coding:");

		if (ImGui::ColorEdit4("Minimum", (float*)& spaceTimeOptions->minColor))
		{
			updatefluctuation = true;
		}
		if (ImGui::InputFloat("Min Value", (float*)& spaceTimeOptions->min_val, 0.1f))
		{
			updatefluctuation = true;
		}

		if (ImGui::ColorEdit4("Maximum", (float*)& spaceTimeOptions->maxColor))
		{
			updatefluctuation = true;
		}

		if (ImGui::InputFloat("Max Value", (float*)& spaceTimeOptions->max_val, 0.1f))
		{
			updatefluctuation = true;
		}

		ImGui::Separator();



		if (ImGui::InputInt("wall-normal", &spaceTimeOptions->wallNoramlPos, 1, 5))
		{
			this->updatefluctuation = true;
		}
		if (ImGui::InputInt("time-position", &spaceTimeOptions->timePosition, 1, 5))
		{
			this->updatefluctuation = true;
		}


		if (ImGui::Checkbox("Absolute Value", &spaceTimeOptions->usingAbsolute))
		{
			this->updatefluctuation = true;
		}


		if (ImGui::DragFloat("height scale", &spaceTimeOptions->height_scale,0.001f,0,0.14f))
		{

			if (spaceTimeOptions->height_scale > 0.0f)
				this->spaceTimeOptions->shading = true;
			this->updatefluctuation = true;
			

		}

		if (ImGui::DragFloat("height offset", &spaceTimeOptions->offset, 0.01f, 0, 10.0f))
		{
			this->updatefluctuation = true;

		}

		if (ImGui::DragFloat("height clamp", &spaceTimeOptions->heightLimit, 0.01f, 0, 5.0f))
		{
			this->updatefluctuation = true;

		}

		if (ImGui::DragFloat("Sampling Rate 0", &spaceTimeOptions->samplingRate_0, 0.00001f, 0.00001f, 1.0f, "%.5f"))
		{
			if (spaceTimeOptions->samplingRate_0 < 0.0001f)
			{
				spaceTimeOptions->samplingRate_0 = 0.0001f;
			}

			this->updatefluctuation = true;
		}

		ImGui::End();
	}
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


void RenderImGuiOptions::drawVisitationMapOptions()
{
	if (b_drawVisitationOptions)
	{
		ImGui::Begin("Visitation Map");
		// Show Cross Sections
		if (ImGui::Checkbox("Show Visitation Map", &this->showVisitationMap))
		{

		}

		if (ImGui::DragFloat("Visitation Threshold", &visitationOptions->visitationThreshold, 0.01f,0.0f,400.0f))
		{
			this->updateRaycasting = true;
			this->updateVisitationMap = true;
		}

		if (ImGui::DragFloat("Raycasting Threshold", &visitationOptions->threshold, 0.1f))
		{
			this->updateRaycasting = true;

		}

		if (ImGui::DragFloat("Amplitude", &visitationOptions->amplitude, 0.001f))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
		}

		if (ImGui::DragFloat("Relaxation", &visitationOptions->relax, 0.1f,0.01f,10000,"%.5f"))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
		
		}

		if (ImGui::DragFloat("Opacity Offset", &visitationOptions->opacityOffset, 0.01f, 0.0f, 1.0f, "%.4f"))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;

		}

		if (ImGui::SliderInt("Ensemble Member", &visitationOptions->ensembleMember, solverOptions->firstIdx, solverOptions->lastIdx))
		{
			visitationOptions->updateMember = true;
		}

		if (updateVisitationMap)
		{
			if (ImGui::Button("Apply Update"))
			{
				this->visitationOptions->applyUpdate = true;
			}
		}
			
		if (ImGui::Combo("Visitation Field", &visitationOptions->visitationField, VisitationField::VisitationFiledList, VisitationField::VisitationField::COUNT))
		{
			this->updateRaycasting = true;
			this->updateTimeSpaceField = true;
		}

		if (ImGui::Checkbox("Save Visitation", &visitationOptions->saveVisitation))
		{

		}

		if (ImGui::Checkbox("auto-update", &visitationOptions->autoUpdate))
		{

		}

		if (ImGui::Checkbox("2D Visiatation", &visitationOptions->visitation2D))
		{

		}

		if (ImGui::SliderInt("selected layer", &visitationOptions->visitationPlane, 0, solverOptions->gridSize[2]))
		{
		}

		ImGui::End();
	}

}


void RenderImGuiOptions::drawImguiOptions()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("View Options");

	if (ImGui::Checkbox("View Solver Options", &this->b_drawSolverOptions))
	{

	}

	if (ImGui::Checkbox("View Raycasting", &this->b_drawRaycastingOptions))
	{

	}

	if (ImGui::Checkbox("View Rendering Options", &this->b_drawLineRenderingOptions))
	{

	}

	if (ImGui::Checkbox("View Time-Space Rendering", &this->b_drawTimeSpaceOptions))
	{

	}

	if (ImGui::Checkbox("View Visitation Options", &this->b_drawVisitationOptions))
	{

	}

	if (ImGui::Checkbox("View Log", &this->b_drawLog))
	{

	}

	ImGui::End();
}


void RenderImGuiOptions::drawDataset()
{
	
	ImGui::Begin("Datasets");
	

	/*if (ImGui::Combo("Dataset 0", reinterpret_cast<int*>(&this->dataset_0),Dataset::datasetList, Dataset::Dataset::COUNT))
	{
		this->updateStreamlines = true;
		this->updatePathlines = true;
		this->updateStreaklines = true;
		this->updateRaycasting = true;

		this->solverOptions->fileChanged = true;
		solverOptions->loadNewfile = true;


		switch (dataset_0)
		{

		case Dataset::Dataset::GRAND_ENSEMBLE_OF_VIS_262:
			this->fieldOptions[1].setField("OF_temperature_dt_vis_262_", "F:\\Grand_ensemble\\", 651, 716, 65, 2.0f, 2.0f, 2.0f, 1, 23, 0.001f);
			break;

			case Dataset::Dataset::KIT2REF_COMP:
			{

				
				this->solverOptions->fileName = "Comp_FieldP";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT2\\Comp_Ref\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 7000000;
				break;			
			}

			case Dataset::Dataset::KIT2OW_COMP:
			{
				this->solverOptions->fileName = "Comp_FieldP";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT2\\Comp_OW\\";
				
				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 7000000;
				break;			
			}
			case Dataset::Dataset::KIT2BF_COMP:
			{
				this->solverOptions->fileName = "FieldP";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT2\\Comp_BF\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				break;			
			}


			case Dataset::Dataset::KIT2OW_OF_STREAM:
			{
				this->solverOptions->fileName = "Comp_OF_";
				this->solverOptions->filePath = "G:\\KIT2\\Comp_OW_OF_Streamwise\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 9000000;
				break;
			}

			case Dataset::Dataset::KIT2OW_OF_LAMBDA:
			{
				this->solverOptions->fileName = "Comp_OF_";
				this->solverOptions->filePath = "G:\\KIT2\\Comp_OW_OF_Lambda\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->solverOptions->seedBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 7.854f, 2.0f, 3.1415f);
				setArray<int>(&this->solverOptions->gridSize[0], 192, 192, 192);

				this->solverOptions->compressed = true;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->maxSize = 9000000;
				break;
			}




			case Dataset::Dataset::KIT3_FLUC:
			{
				this->solverOptions->fileName = "FieldP";
				this->solverOptions->filePath = "F:\\KIT3\\Fluc\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64,503,2048);
				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 1000;


				this->solverOptions->dt = 0.001f;
				break;

			}

			case Dataset::Dataset::KIT3_INITIAL_COMPRESSED:
			{
				this->solverOptions->fileName = "initialComp_";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT3\\Comp_initial\\";





				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 1000;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.00012f;
				this->solverOptions->periodic = true;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 69000000;
				break;

			}

		



			case Dataset::Dataset::KIT3_OF_AVG50_COMPRESSED:
			{
				this->solverOptions->fileName = "OF_AVG_COMP_50_";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT3\\Comp_OF_AVG50\\";


				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 949;
				this->solverOptions->currentIdx = 551;

				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);


				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 96000000;

				break;

			}			case Dataset::Dataset::KIT3_OF_ENERGY_COMPRESSED:
			{
				this->solverOptions->fileName = "OF_Energy_";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT3\\Comp_OF_Energy\\";


				this->solverOptions->firstIdx = 551;
				this->solverOptions->lastIdx = 949;
				this->solverOptions->currentIdx = 551;

				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);


				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 125000000;

				break;

			}
			case Dataset::Dataset::KIT3_OF_COMPRESSED:
			{
				this->solverOptions->fileName = "Comp_OF_";
				this->solverOptions->filePath = "Y:\\KIT3\\Comp_OF\\";


				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 999;
				this->solverOptions->currentIdx = 500;

				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);


				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 46000000;

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
				this->solverOptions->compressed = true;
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
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 70000000;

				break;

			}

			case Dataset::Dataset::KIT3_AVG_COMPRESSED_50:
			{
				this->solverOptions->fileName = "Field_AVG_Comp_";
				this->solverOptions->filePath = "G:\\Dataset_Compressed\\KIT3\\Comp_TimeAVG50\\";


				setArray<float>(&this->solverOptions->gridDiameter[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 0.4f, 2.0f, 7.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 0.4f, 2.0f, 7.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 64, 503, 2048);

				this->solverOptions->firstIdx = 500;
				this->solverOptions->lastIdx = 900;
				this->solverOptions->currentIdx = 500;
				this->solverOptions->dt = 0.001f;
				this->solverOptions->periodic = true;
				this->solverOptions->compressed = true;
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
				this->solverOptions->compressed = true;
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
				this->solverOptions->compressed = false;

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
				this->solverOptions->compressed = false;

				break;

			}

			case Dataset::Dataset::RBC:
			{
				this->solverOptions->fileName = "Field";
				this->solverOptions->filePath = "E:\\TUI_RBC_Small\\tui_ra1e5\\";



				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 0.2f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 0.2f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 0.2f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);



				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->compressed = false;

				break;

			}	
			case Dataset::Dataset::RBC_VIS_OF:
			{
				this->solverOptions->fileName = "Field";
				this->solverOptions->filePath = "G:\\RBC_Vis\\";



				setArray<float>(&this->solverOptions->gridDiameter[0], 5.0f, 0.2f, 5.0f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.0f, 0.2f, 5.0f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.0f, 0.2f, 5.0f);
				setArray<int>(&this->solverOptions->gridSize[0], 1024, 32, 1024);



				this->solverOptions->dt = 0.01f;
				this->solverOptions->periodic = false;
				this->solverOptions->compressed = false;

				break;

			}


			case Dataset::Dataset::KIT4_L1:
			{

				this->solverOptions->fileName = "KIT4_mipmap";
				this->solverOptions->filePath = "Y:\\KIT4\\TimeResolved\\Initial\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<int>(&this->solverOptions->gridSize[0], 1023, 124, 1024);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 20;
				this->solverOptions->compressed = false;

				break;
			}
			case Dataset::Dataset::KIT4_L1_FLUCTUATION:
			{

				this->solverOptions->fileName = "KIT4_mipmap_fluc_";
				this->solverOptions->filePath = "Y:\\KIT4\\TimeResolved\\Fluctuation\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<int>(&this->solverOptions->gridSize[0], 1023, 124, 1024);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 20;
				this->solverOptions->compressed = false;

				break;
			}
			case Dataset::Dataset::KIT4_L1_TIME_AVG_20:
			{

				this->solverOptions->fileName = "KIT4_mipmap_avg20_";
				this->solverOptions->filePath = "Y:\\KIT4\\TimeResolved\\time-averaged_20\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<int>(&this->solverOptions->gridSize[0], 1023, 124, 1024);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 20;
				this->solverOptions->compressed = false;

				break;
			}			
			case Dataset::Dataset::KIT4_L1_TIME_AVG_20_FLUC:
			{

				this->solverOptions->fileName = "KIT4_mipmap_fluc_avg20_";
				this->solverOptions->filePath = "Y:\\KIT4\\TimeResolved\\time-averaged_20_Fluc\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<int>(&this->solverOptions->gridSize[0], 1023, 124, 1024);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 20;
				this->solverOptions->compressed = false;

				break;
			}	
			case Dataset::Dataset::KIT4_L1_INITIAL_COMP:
			{

				this->solverOptions->fileName = "KIT4_mipmap_Comp";
				this->solverOptions->filePath = "G:\\KIT4\\Initial_Comp\\";

				setArray<float>(&this->solverOptions->gridDiameter[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->solverOptions->seedBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<float>(&this->raycastingOptions->clipBox[0], 5.2409267f, .2f, 2.51204753716f);
				setArray<int>(&this->solverOptions->gridSize[0], 1023, 124, 1024);

				this->solverOptions->dt = 0.001f;
				this->solverOptions->firstIdx = 1;
				this->solverOptions->lastIdx = 32;
				this->solverOptions->compressed = true;
				this->solverOptions->maxSize = 250000000;

				break;
			}






		}
	}*/



	

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