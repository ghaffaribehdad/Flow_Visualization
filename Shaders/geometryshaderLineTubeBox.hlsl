
cbuffer GS_CBuffer
{
	float4x4 View;
	float4x4 Proj;
	float3 viewDir;
	float tubeRadius;
	float3 eyePos;
	float size;
	float4 color;
};




struct GS_INPUT
{
	float3 inPosition : POSITION;
	float3 inTangent: TANGENT;
	unsigned int inLineID : LINEID;
	float inMeasure : MEASURE;
	float3 inNormal : NORMAL;
};


struct GS_OUTPUT
{

	float4 outPosition : SV_POSITION;
	float3 outTangent : TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
};




// The plane is to calculate vertices and then transform them in the camera coordinate
[maxvertexcount(18)]
void main(line GS_INPUT input[2], inout TriangleStream<GS_OUTPUT> output)
{


	if (input[0].inLineID == input[1].inLineID)
	{
		GS_OUTPUT vertex0;
		GS_OUTPUT vertex1;


		// calculates ray direction from eye to the vertex
		float3 viewDirection = normalize(input[0].inPosition - eyePos);

		// A vector in the cross-section plane
		float3 orient0 = normalize(cross(input[0].inTangent, viewDirection));
		float3 orient1 = normalize(cross(input[1].inTangent, viewDirection));

		// Radius of the tubes
		float tubeRad = tubeRadius;

		// in degree
		float angle = 360.0f / 8.0f;
		// in radian
		angle = (angle / 180.0f) * 3.14159265359f;

		// pre-computing the sine and cosine0
		float sine = sin(angle);
		float cosine = cos(angle);

		float3 orient0_rotated = orient0;
		float3 orient1_rotated = orient1;

		for (int i = 0; i < 9; i++)
		{
			// rotate the orientation vector around normal for 45 degree (input[0])
			orient0_rotated = orient0_rotated * cosine + cross(input[0].inTangent, orient0_rotated) * sine;


			// rotate the orientation vector around normal for 45 degree (input[1])
			orient1_rotated = orient1_rotated * cosine + cross(input[1].inTangent, orient1_rotated) * sine;


			float3 position0 = input[0].inPosition + orient0_rotated * tubeRad;
			float3 position1 = input[1].inPosition + orient1_rotated * tubeRad;

			if (input[0].inLineID == 1) // Vertical Line
			{
				position0.y += .01;
				position1.y -= .01;
			}


			// SV_POSITION
			vertex0.outPosition = mul(View,float4(position0,1.0f));
			vertex0.outPosition = mul(Proj,vertex0.outPosition );

			vertex1.outPosition = mul(View, float4(position1, 1.0f));
			vertex1.outPosition = mul(Proj, vertex1.outPosition );

			//vertex1.outPosition = mul(float4(position1, 1.0f), transpose(View));
			//vertex1.outPosition = mul(vertex1.outPosition, transpose(Proj));


			// Normals
			if (input[0].inLineID == 0)
			{
				vertex0.outNormal = -orient0_rotated;
				vertex1.outNormal = -orient1_rotated;
			}
			else // To Do: Change the direction of the normal for sides
			{
				vertex0.outNormal = -orient0_rotated;
				vertex1.outNormal = -orient1_rotated;
			}
			
			
			// Colors
			vertex0.outMeasure = input[0].inMeasure;
			vertex1.outMeasure = input[0].inMeasure;

			// Tangent
			vertex0.outTangent = input[0].inTangent;
			vertex1.outTangent = input[1].inTangent;

			// World Position
			vertex0.outLightDir = viewDir;
			vertex1.outLightDir = viewDir;

			output.Append(vertex0);
			output.Append(vertex1);

		}

		output.RestartStrip();
	}		
}



