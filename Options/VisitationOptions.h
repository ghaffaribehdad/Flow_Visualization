#pragma once
#include "RaycastingOptions.h"
struct VisitationOptions
{

	bool initialized = false;
	bool resize = false;
	bool applyUpdate = false;
	bool autoUpdate = false;

	float amplitude = 1.0f;
	float relax = 1.0f;
	float opacityOffset = 0.1f;


	float threshold = 0.01f;
	float visitationThreshold = 220;

	int ensembleMember = 10;
	int visitationField = 0;
	int visitationPlane = 0;

	int raycastingMode = Visitation::RaycastingMode::DVR_VISITATION;
	bool updateMember = false;
	bool saveVisitation = false;
	bool visitation2D = false;
};

namespace VisitationField
{
	static const char* const VisitationFiledList[] =
	{
		"Visitation",
		"Mean visitation",
		"Mean value",
		"none"
	};
	enum VisitationField
	{
		VISITATION,
		MEAN_VISITATION,
		MEAN_VALUE,
		NONE,
		COUNT
	};
}