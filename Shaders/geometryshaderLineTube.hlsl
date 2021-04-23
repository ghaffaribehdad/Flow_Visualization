
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
	float3 outTangent : TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
	float transparency : TRANSPARENCY;
};



// The plane is to calculate vertices and then transform them in the camera coordinate
[maxvertexcount(18)]
void main(lineadj GS_INPUT input[4], inout TriangleStream<GS_OUTPUT> output)
{

	bool append = true;

	float transparency = 1;

	switch (transparencyMode)
	{
	case(0):
	{
		transparency = 1 - abs(input[1].inPosition.x - streakPos) / gridDiameter.x;

		break;
	}
	case(1):
	{
		transparency = 1 - abs(currentTime - (float)input[1].inTime) / (float)timDim;

		break;
	}

	}

	float3 pos0 = input[0].inPosition - (gridDiameter / 2);
	float3 pos1 = input[1].inPosition - (gridDiameter / 2);
	float3 pos2 = input[2].inPosition - (gridDiameter / 2);
	float3 pos3 = input[3].inPosition - (gridDiameter / 2);

	switch (projection)
	{
	case(1):
	{
		pos0.x = input[0].inInitial.x - (gridDiameter.x / 2);
		pos1.x = input[1].inInitial.x - (gridDiameter.x / 2);
		pos2.x = input[2].inInitial.x - (gridDiameter.x / 2);
		pos3.x = input[3].inInitial.x - (gridDiameter.x / 2);

		break;
	}
	case(2):
	{
		pos0.y = input[1].inInitial.y - (gridDiameter.y / 2);
		pos1.y = input[1].inInitial.y - (gridDiameter.y / 2);
		pos2.y = input[1].inInitial.y - (gridDiameter.y / 2);
		pos3.y = input[1].inInitial.y - (gridDiameter.y / 2);
		break;
	}
	case(3):
	{
		pos0.z = input[1].inInitial.z - (gridDiameter.z / 2);
		pos1.z = input[1].inInitial.z - (gridDiameter.z / 2);
		pos2.z = input[1].inInitial.z - (gridDiameter.z / 2);
		pos3.z = input[1].inInitial.z - (gridDiameter.z / 2);
		break;
	}
	case(4):
	{
		pos0.x = particlePlanePos;
		pos1.x = particlePlanePos;
		pos2.x = particlePlanePos;
		pos3.x = particlePlanePos;
		
		break;
	}

	default:
		break;
	}


	// Filter periodicity
	if (!periodicity)
	{
		if (abs(pos2.x) > gridDiameter.x / 2 || abs(pos2.y) > gridDiameter.y / 2 || abs(pos2.z) > gridDiameter.z / 2)
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

	
	if (input[1].inLineID == input[2].inLineID && append)
	{




		GS_OUTPUT vertex0;
		GS_OUTPUT vertex1;



		// Radius of the tubes
		float tubeRad = tubeRadius;

		// in degree
		float angle = 360.0f / 8.0f;
		// in radian
		angle = (angle / 180.0f) * 3.14159265359f;

		// pre-computing the sine and cosine
		float sine = sin(angle);
		float cosine = cos(angle);
		float3 tangent0 = { 0,0,0 };
		float3 tangent1 = { 0,0,0 };

		// for the first line segment (the line ID is different for the first two)
		if (input[0].inLineID != input[1].inLineID)
		{
			tangent0 = normalize(pos2 - pos1);
			tangent1 = normalize(normalize(pos2 - pos1) + normalize(pos3 - pos2));
		}

		// for the last line segment (the line ID is different for the third and fourth vertex)
		else if (input[3].inLineID != input[2].inLineID)
		{
			tangent0 = normalize(normalize(pos1 - pos0) + normalize(pos2 - pos1));
			tangent1 = normalize(normalize(pos2 - pos1));
		}
		else
		{
			tangent0 = normalize(normalize(pos1 - pos0) + normalize(pos2 - pos1));
			tangent1 = normalize(normalize(pos2 - pos1) + normalize(pos3 - pos2));

		}

		float3 tangent = normalize(tangent0 + tangent1); // average of the tangents
		float3 vecNormal1 = { 0,1.0f,0};
		float3 vecNormal2 = vecNormal1;

		float3 orient0_rotated = normalize(cross(vecNormal1, tangent0));
		float3 orient1_rotated = normalize(cross(vecNormal2, tangent1));

		for (int i = 0; i < 9; i++)
		{
			// rotate the orientation vector around tangent for 45 degree (input[0])
			orient0_rotated = orient0_rotated * cosine + cross(tangent0, orient0_rotated) * sine;


			// rotate the orientation vector around tangent for 45 degree (input[1])
			orient1_rotated = orient1_rotated * cosine + cross(tangent1, orient1_rotated) * sine;


			float3 position0 = pos1 + orient0_rotated * tubeRad;
			float3 position1 = pos2 + orient1_rotated * tubeRad;

			// SV_POSITION
			// SV_POSITION
			vertex0.outPosition = mul(View, float4(position0, 1.0f));
			vertex0.outPosition = mul(Proj, vertex0.outPosition);

			vertex1.outPosition = mul(View, float4(position1, 1.0f));
			vertex1.outPosition = mul(Proj, vertex1.outPosition);

			vertex0.transparency = transparency;
			vertex1.transparency = transparency;


			// Normals
			vertex0.outNormal = -orient0_rotated;
			vertex1.outNormal = -orient1_rotated;

			// Colors
			vertex0.outMeasure = (input[1].inMeasure + input[2].inMeasure) * 0.5f;
			vertex1.outMeasure = (input[1].inMeasure + input[2].inMeasure) * 0.5f;

			// Tangent
			vertex0.outTangent = tangent;
			vertex1.outTangent = tangent;

			// World Position
			vertex0.outLightDir = viewDir;
			vertex1.outLightDir = viewDir;

			if (usingThreshold)
			{
				if (abs(input[1].inPosition.x - streakPos) < threshold)
				{
					output.Append(vertex0);
					output.Append(vertex1);
				}
			}
			else
			{
				output.Append(vertex0);
				output.Append(vertex1);
			}


		}



		output.RestartStrip();
	}

}



