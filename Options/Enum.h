#pragma once

enum SeedingPattern
{
	SEED_RANDOM = 0,
	SEED_GRIDPOINTS,
	SEED_FILE,
};



enum IsoMeasure
{
	VelocityMagnitude = 0,
	Velocity_x,
	Velocity_y,
	Velocity_Z,
	ShearStress,
	Position_Y
};


static const char* const IsoMeasureModes[] = 
{ 
	"Velocity Magnitude",
	"Velocity X",
	"Velocity Y",
	"Velocity Z",
	"Shear Stress"
};
static const char* const ColorModeList[] = {"V", "Vx", "Vy", "Vz"};

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
	"Grid Points"
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



//enum Projection
//{
//	NO_PROJECTION =0,
//	ZY_PROJECTION,
//	XZ_PROJECTION,
//	XY_PROJECTION
//};