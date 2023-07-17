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
	"Double",
};





namespace Dataset
{
	enum Dataset
	{
		NONE = 0,
		KIT2REF_COMP,
		KIT2REF,
		KIT2OW_COMP,
		KIT2BF_COMP,
		KIT2OW_OF_STREAM,
		KIT2OW_OF_LAMBDA,
		KIT3_RAW,
		KIT3_FLUC,
		KIT3_INITIAL_COMPRESSED,
		KIT3_COMPRESSED,
		KIT3_OF_AVG50_COMPRESSED,
		KIT3_OF_ENERGY_COMPRESSED,
		KIT3_OF_COMPRESSED,
		KIT3_OF_COMPRESSED_FAST,
		KIT3_AVG_COMPRESSED_10,
		KIT3_AVG_COMPRESSED_50,
		KIT3_SECONDARY_COMPRESSED,
		KIT3_OF,
		KIT3_TIME_SPACE_1000_TZY,
		KIT3_TIME_SPACE_1000_TYX,
		RBC,
		RBC_VIS_OF,
		RBC_AVG,
		RBC_AVG_20,
		RBC_AVG_50,
		RBC_AVG_100,
		RBC_AVG_OF_20,
		RBC_AVG_OF_50,
		RBC_AVG_OF_100,
		RBC_OF,
		RBC_AVG_OF_600,
		RBC_AVG500,
		EJECTA,
		EJECTA_CUSZ,
		EJECTA_CC,
		MESHKOV,
		MIRANDA,
		KIT4_L1,
		KIT4_L1_FLUCTUATION,
		KIT4_L1_TIME_AVG_20,
		KIT4_L1_TIME_AVG_20_FLUC,
		KIT4_L1_INITIAL_COMP,
		TUM,
		TUM_MEAN_REAMOVED,
		TUM_L1,
		TUM_L2,
		SMOKE_00050_ENSEMBLE,
		GRAND_ENSEMBLE_TIME,
		GRAND_ENSEMBLE_OF_VIS_262,
		GRAND_ENSEMBLE_OF_VIS_263,
		GRAND_ENSEMBLE_OF_VIS_264,
		GRAND_ENSEMBLE_OF_AVG_262,
		GRAND_ENSEMBLE_OF_AVG_263,
		GRAND_ENSEMBLE_OF_AVG_264,
		MUTUAL_INFO,
		MUTUAL_INFO_1,
		KIT3_CC,
		KIT3_CUSZ,
		COUNT,
	};

	static const char* datasetList[]
	{
		"None",
		"KIT2 Ref Compressed",
		"KIT2 Ref",
		"KIT2 Oscillating Wall Compressed",
		"KIT2 Virtual Body Compressed",
		"KIT2 OW OF Stream Compressed",
		"KIT2 OW OF Lambda Compressed",
		"KIT3 Initi",
		"KIT3 Fluctuation",
		"KIT3 Inital Comp.",
		"KIT3 Fluc. Comp.",
		"KIT3 OF AVG50 Comp.",
		"KIT3 OF Energy Comp.",
		"KIT3 OF Comp.",
		"KIT3 OF Comp. Fast",
		"KIT3 AVG Comp. 10",
		"KIT3 AVG Comp. 50",
		"KIT3 Secondary Comp.",
		"KIT3 OF ",
		"KIT3 Time-Space TZY",
		"KIT3 Time-Space TYX",
		"RBC",
		"RBC Vis OF",
		"RBC_AVG",
		"RBC_AVG_20",
		"RBC_AVG_50",
		"RBC_AVG_100",
		"RBC_AVG_OF_20",
		"RBC_AVG_OF_50",
		"RBC_AVG_OF_100",
		"RBC OF",
		"RBC_AVG_OF_600",
		"RBC_AVG500",
		"Ejecta",
		"Ejecta CUSZ",
		"Ejecta CC",
		"Meshkov",
		"Miranda",
		"KIT4 L1",
		"KIT4 L1 Fluc",
		"KIT4 L1 TimeAvg 20",
		"KIT4 L1 TimeAvg 20 Fluc",
		"KIT4 L1 Initial Comp.",
		"TUM",
		"TUM Fluc",
		"TUM L1",
		"TUM L2",
		"Smoke Ensemble 50",
		"Grand Ensemble Time",
		"Grand Ensemble OF vis 262",
		"Grand Ensemble OF vis 263",
		"Grand Ensemble OF vis 264",
		"Grand Ensemble OF avg 262",
		"Grand Ensemble OF avg 263",
		"Grand Ensemble OF avg 264",
		"Mutual Information",
		"Mutual Information 1",
		"KIT3 CC",
		"KIT3 CUSZ",
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
	"lambda2",
	"Velocity X Planar",
	"Velocity Y Planar",
	"Velocity Z Planar"
};

namespace ColorMode
{
	static const char* const ColorModeList[] = 
	{ 
		"Velocity Magnitude",
		"u", 
		"v",
		"w",
		"Curvature",
		"Distance to Projection Plane"
	};
	enum ColorMode
	{
		VELOCITY_MAG,
		U_VELOCITY,
		V_VELOCITY,
		W_VELOCITY,
		CURVATURE,
		DISTANCE_STREAK,
		COUNT,
	};
}

namespace LineRenderingMode
{
	static const char* const LineRenderingModeList[] =
	{
		"Streamlines",
		"Pathlines",
		"Streaklines",
	};
	enum LineRenderingMode
	{
		STREAMLINES,
		PATHLINES,
		STREAKLINES,
		COUNT,
	};
}


namespace ActiveField
{
	static const char* const ActiveFieldList[] =
	{
		"0",
		"1",
		"2",
		"3",
	};
}
namespace VisitationMode
{
	static const char* const VisitationModeList[] =
	{
		"Mean-diff"
	};
	enum VisitationMode
	{
		MEAN_DIFF = 0,
		COUNT,
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
	"Quadrant Dev",
};


static const char* const CrossSectionMode[] =
{
	"XY",
	"XZ",
	"YZ",
};


static const char* const spanMode[] =
{
	"Wall-Normal",
	"Time",
	"3D Vol",
};



static const char* const ColorCode_fluctuationField[] =
{
	"None",
	"Vx_fluctuation",
	"Vy_fluctuation",
	"Vz_fluctuation",
};

static const char* const SeedPatternList[] = 
{
	"Random",
	"Grid Points",
	"Tilted Plane",
};





static const char* const FieldMode[] =
{
	"ff (Vx,Vy)",
	"ff (Vz,Vy)",
	"fi (Vx,Vy)",
};

static const char* const heightMode[] =
{
	"Height",
	"FTLE",
};


namespace Projection
{
	static const char* const ProjectionList[] =
	{
		"None",
		"ZY-Plane",
		"XZ-Plane",
		"XY-Plane",
		"Streak Proj.",
		"Streak Proj. Fix",
	};

	enum Projection
	{
		NO_PROJECTION = 0,
		ZY_PROJECTION,
		XZ_PROJECTION,
		XY_PROJECTION,
		STREAK_PROJECTION,
		STREAK_PROJECTION_FIX,
		COUNT,
	};
}
