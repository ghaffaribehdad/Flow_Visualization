#pragma once

enum SeedingPattern
{
	SEED_RANDOM = 0,
	//SEED_REGULAR,
	//SEED_FILE,
};

enum IntegrationMethod
{
	EULER_METHOD = 0,
	//MODIFIED_EULER,
	//RK4_METHOD,
	//RK5_METHOD,
};



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