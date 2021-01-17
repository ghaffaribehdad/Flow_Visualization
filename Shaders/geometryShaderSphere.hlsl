cbuffer GS_CBuffer
{
	float4x4 View;
	float4x4 Proj;
	float3 viewDir;
	float tubeRadius;
	float3 eyePos;
	int projection;
	float3 gridDiameter;
	bool periodicity;
};




struct GS_INPUT
{
	float3 inPosition : POSITION;
	float3 inTangent: TANGENT;
	unsigned int inLineID : LINEID;
	float inMeasure : MEASURE;
	float3 inNormal : NORMAL;
	float3 inInitial : INITIALPOS;
};



struct GS_OUTPUT
{

	float4 outPosition : SV_POSITION;
	float3 outCenter : CENTER;
	float3 outViewPos: POS;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
	float radius : RADIUS;
};




// The plane is to calculate vertices and then transform them in the camera coordinate
[maxvertexcount(18)]
void main(point GS_INPUT input[1], inout TriangleStream<GS_OUTPUT> output)
{


	float3 pos0 = input[0].inPosition - (gridDiameter / 2);


	switch (projection)
	{
	case(1):
	{
		pos0.x = input[0].inInitial.x - (gridDiameter.x / 2);

		break;
	}
	case(2):
	{
		pos0.y = input[0].inInitial.y - (gridDiameter.y / 2);

		break;
	}
	case(3):
	{
		pos0.z = input[0].inInitial.z - (gridDiameter.z / 2);

		break;
	}
	default:
		break;
	}


	switch (periodicity)
	{
	case(false): // not periodic
	{
		if (abs(pos0.x) > gridDiameter.x / 2 || abs(pos0.y) > gridDiameter.y / 2 || abs(pos0.z) > gridDiameter.z / 2)
		{
			output.RestartStrip();
		}

		break;
	}
	case(true): // periodic
	{

		break;
	}
	}


	GS_OUTPUT vertex0;

	float3 tangent0 = input[0].inTangent;
	float3 inPlaneVec = normalize(cross(viewDir, tangent0));
	float3 inPlaneVecPer = normalize(cross(inPlaneVec, viewDir));



	vertex0.outCenter = pos0;

	vertex0.outLightDir = normalize(-tangent0);
	vertex0.outNormal = normalize(-tangent0);;
	vertex0.outMeasure = input[0].inMeasure;

	vertex0.radius = tubeRadius;

	vertex0.outPosition = mul(View, float4(pos0  + tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = pos0 + tubeRadius * inPlaneVec;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(pos0 + tubeRadius * inPlaneVecPer, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = pos0 + tubeRadius * inPlaneVecPer;
	output.Append(vertex0);

	vertex0.outPosition = mul(View, float4(pos0 - tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = pos0 - tubeRadius * inPlaneVec;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(pos0 - tubeRadius * inPlaneVecPer, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = pos0 - tubeRadius * inPlaneVecPer;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(pos0 + tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = pos0 + tubeRadius * inPlaneVec;
	output.Append(vertex0);



	
	output.RestartStrip();
}



