#pragma once

enum SeedingPattern
{
	SEED_RANDOM = 0,
	SEED_GRIDPOINTS,
	SEED_TILTED_PLANE,
	SEED_FILE,
};





static const char* const HeightfieldRenderingMode[] =
{
	"Single",
	"Double"
};


enum IsoMeasure
{
	VelocityMagnitude = 0,
	Velocity_X,
	Velocity_Y,
	Velocity_Z,
	ShearStress,
	TURBULENT_DIFFUSIVITY,
	Velocity_X_Plane,
	COUNT
};


static const char* const IsoMeasureModes[] = 
{ 
	"Velocity Magnitude",
	"Velocity X",
	"Velocity Y",
	"Velocity Z",
	"Shear Stress",
	"Turbulent Diffusivity",
	"Velocity X Planar"
};
static const char* const ColorModeList[] = {"V", "Vx", "Vy", "Vz","i_Vx","i_Vy","i_Vz"};

static const char* const ColorCode_DispersionList[] = 
{
	"None",
	"Vx_fluctuation",
	"Vy",
	"Vz",
	"Dev_Z",
	"Dev_XZ",
	"Advection Dist.",
	"Advection Dist. Proj.",
	"Quadrant Dev"
};


static const char* const CrossSectionMode[] =
{
	"XY",
	"XZ",
	"YZ"
};

static const char* const spanMode[] =
{
	"Wall-Normal",
	"Time",
	"3D Vol"
};


static const char* const ColorCode_fluctuationField[] =
{
	"None",
	"Vx_fluctuation",
	"Vy_fluctuation",
	"Vz_fluctuation"
};

static const char* const SeedPatternList[] = 
{
	"Random",
	"Grid Points",
	"Tilted Plane"
};

static const char* const ProjectionList[] = 
{ 
	"None",
	"ZY-Plane",
	"XZ-Plane", 
	"XY-Plane"
};



static const char* const FieldMode[] =
{
	"ff (Vx,Vy)",
	"ff (Vz,Vy)",
	"fi (Vx,Vy)"
};

static const char* const heightMode[] =
{
	"Height",
	"FTLE"
};



enum Projection
{
	NO_PROJECTION =0,
	ZY_PROJECTION,
	XZ_PROJECTION,
	XY_PROJECTION
};