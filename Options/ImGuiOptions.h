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


namespace Dataset
{
	enum Dataset
	{
		NONE = 0,
		KIT2REF_COMP,
		KIT2OW,
		KIT2BF,
		KIT3_FLUC,
		KIT3_COMPRESSED,
		KIT3_INITIAL_COMPRESSED,
		KIT3_OF_COMPRESSED,
		RBC,
		RBC_AVG,
		RBC_OF,
		RBC_AVG_OF_600,
		RBC_AVG500,
		TEST_FIELD,
		COUNT
	};

	static const char* datasetList[]
	{
		"None",
		"KIT2 Ref Compressed",
		"KIT2 Oscillating Wall",
		"KIT2 Virtual Body",
		"KIT3 Fluctuation",
		"KIT3 Fluc. Comp.",
		"KIT3 Initial Comp.",
		"KIT3 OF Comp.",
		"RBC",
		"RBC_AVG",
		"RBC OF",
		"RBC_AVG_OF_600",
		"RBC_AVG500",
		"Test Field"
	};

}



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

namespace ColorMode
{
	static const char* const ColorModeList[] = 
	{ 
		"Velocity Magnitude",
		"u", 
		"v",
		"w",
		"Curvature"
	};
	enum ColorMode
	{
		VELOCITY_MAG,
		U_VELOCITY,
		V_VELOCITY,
		W_VELOCITY,
		CURVATURE,
		COUNT
	};
}

namespace LineRenderingMode
{
	static const char* const lineRenderingModeList[] =
	{
		"Streamlines",
		"Pathlines",
		"Streaklines",
	};
	enum lineRenderingMode
	{
		STREAMLINES,
		PATHLINES,
		STREAKLINES,
		COUNT
	};
}


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


namespace Projection
{
	static const char* const ProjectionList[] =
	{
		"None",
		"ZY-Plane",
		"XZ-Plane",
		"XY-Plane",
	};

	enum Projection
	{
		NO_PROJECTION = 0,
		ZY_PROJECTION,
		XZ_PROJECTION,
		XY_PROJECTION,
		COUNT
	};
}
