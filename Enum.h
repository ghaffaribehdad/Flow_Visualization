#pragma once

enum SeedingPattern
{
	SEED_RANDOM = 0,
	SEED_REGULAR,
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

enum InterpolationMethod
{
	Linear,
};

enum Precision
{
	BIT_32,
	BIT_64
};