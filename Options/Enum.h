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
};


static const char* const IsoMeasureModes[] = { "Velocity Magnitude", "Velocity X", "Velocity Y", "Velocity Z","Shear Stress"};
static const char* const ColorModeList[] = {"V", "Vx", "Vy", "Vz"};
static const char* const SeedPatternList[] = { "Random", "Grid Points"};
static const char* const ProjectionList[] = { "None", "ZY-Plane", "XZ-Plane", "XY-Plane" };



enum Projection
{
	NO_PROJECTION =0,
	ZY_PROJECTION,
	XZ_PROJECTION,
	XY_PROJECTION
};