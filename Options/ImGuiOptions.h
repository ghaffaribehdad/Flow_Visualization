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
		KIT2REF,
		KIT2REF_OF_TRUNC,
		KIT2REF_OF_FLUC_TRUNC,
		KIT2REF_OF_PERIODIC,
		KIT2BF,
		KIT2OW,
		KIT3,
		KIT3_COMPRESSED,
		KIT2_COMPRESSED_REF,
		KIT3FAST,
		KIT3_MIPMAP,
		KIT3_OF_MIPMAP,
		MOTIONFIELD_KIT3,
		MOTIONFIELD_KIT3_PERIODIC,
		ENSTROPHY_OF_KIT3,
		COUNT
	};

	static const char* datasetList[]
	{
		"None",
		"KIT2 Ref",
		"KIT2 Ref OF Trunc",
		"KIT2 Ref OF Fluc Trunc",
		"KIT2 Ref OF  PRIODIC",
		"KIT2 Virtual Body",
		"KIT2 Oscillating Wall",
		"KIT3",
		"KIT3 Compressed",
		"KIT2 Compressed Ref",
		"KIT3 Fast",
		"KIT3 MipmapL1",
		"KIT3 Mipmap_of",
		"Motion Field KIT3",
		"Motion Field KIT3 Periodic",
		"Enstrophy OF"
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