
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
	float particlePlanePos;
	unsigned int transparencyMode;
	unsigned int timDim;
	float streakPos;
	unsigned int currentTime;
	bool usingThreshold;
	float threshold;
};




struct GS_INPUT
{
	float3 inPosition : POSITION;
	float3 inTangent: TANGENT;
	unsigned int inLineID : LINEID;
	float inMeasure : MEASURE;
	float3 inNormal : NORMAL;
	float3 inInitial : INITIALPOS;
	unsigned int inTime : TIME;
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
	float transparency : TRANSPARENCY;
};




// The plane is to calculate vertices and then transform them in the camera coordinate
[maxvertexcount(18)]
void main(point GS_INPUT input[1], inout TriangleStream<GS_OUTPUT> output)
{

	bool append = true;



	float transparency = 1;

	float3 pos0 = input[0].inPosition - (gridDiameter / 2);
	
	switch (transparencyMode)
	{
	case(0): // Based on position
	{
		transparency = 1 - abs(input[0].inPosition.x - streakPos) / gridDiameter.x;
		break;
	}
	case(1): // Based on Time
	{
		transparency = 1 - abs(currentTime - (float)input[0].inTime) / (float)timDim;
		break;
	}

	}


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
	case(4):
	{
		pos0.x = particlePlanePos;
		break;
	}

	}

	if (!periodicity)
	{
		if (abs(pos0.x) > gridDiameter.x / 2 || abs(pos0.y) > gridDiameter.y / 2 || abs(pos0.z) > gridDiameter.z / 2)
		{
			append = false;
		}
	}

	if (usingThreshold)
	{
		if (abs(input[0].inPosition.x - streakPos) > threshold)
		{
			append = false;
		}
	}

	if (append)
	{
		GS_OUTPUT vertex0;

		float3 tangent0 = input[0].inTangent;
		float3 inPlaneVec = normalize(cross(viewDir, tangent0));
		float3 inPlaneVecPer = normalize(cross(inPlaneVec, viewDir));


		vertex0.transparency = transparency;
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

	}

	output.RestartStrip();
}



