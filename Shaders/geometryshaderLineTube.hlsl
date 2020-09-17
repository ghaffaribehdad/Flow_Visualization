
cbuffer GS_CBuffer
{
	float4x4 View;
	float4x4 Proj;
	float3 viewDir;
	float tubeRadius;
	float3 eyePos;
	float size;

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
void main(lineadj GS_INPUT input[4], inout TriangleStream<GS_OUTPUT> output)
{


	if (input[1].inLineID == input[2].inLineID)
	{
		GS_OUTPUT vertex0;
		GS_OUTPUT vertex1;



		float3 vecNormal1 = input[1].inNormal;
		float3 vecNormal2 = input[1].inNormal;

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
			tangent0 = normalize(input[2].inPosition - input[1].inPosition);
			tangent1 = normalize(normalize(input[2].inPosition - input[1].inPosition) + normalize(input[3].inPosition - input[2].inPosition));
		}

		// for the last line segment (the line ID is different for the third and fourth vertex)
		else if(input[3].inLineID != input[2].inLineID)
		{
			tangent0 = normalize(normalize(input[1].inPosition - input[0].inPosition) + normalize(input[2].inPosition - input[1].inPosition));
			tangent1 = normalize(normalize(input[2].inPosition - input[1].inPosition));
		}
		else
		{
			tangent0 = normalize(normalize(input[1].inPosition - input[0].inPosition) + normalize(input[2].inPosition - input[1].inPosition));
			tangent1 = normalize(normalize(input[2].inPosition - input[1].inPosition) + normalize(input[3].inPosition - input[2].inPosition));		
	
		}

		float3 tangent = normalize(tangent0 + tangent1);


		float3 orient0_rotated = normalize(cross(vecNormal1, tangent0));
		float3 orient1_rotated = normalize(cross(vecNormal2, tangent1));

		for (int i = 0; i < 9; i++)
		{
			// rotate the orientation vector around tangent for 45 degree (input[0])
			orient0_rotated = orient0_rotated * cosine + cross(tangent0, orient0_rotated) * sine;


			// rotate the orientation vector around tangent for 45 degree (input[1])
			orient1_rotated = orient1_rotated * cosine + cross(tangent1, orient1_rotated) * sine;


			float3 position0 = input[1].inPosition + orient0_rotated * tubeRad;
			float3 position1 = input[2].inPosition + orient1_rotated * tubeRad;

			// SV_POSITION
			// SV_POSITION
			vertex0.outPosition = mul(View, float4(position0, 1.0f));
			vertex0.outPosition = mul(Proj, vertex0.outPosition);

			vertex1.outPosition = mul(View, float4(position1, 1.0f));
			vertex1.outPosition = mul(Proj, vertex1.outPosition);



			// Normals
			vertex0.outNormal = -orient0_rotated;
			vertex1.outNormal = -orient1_rotated;
			
			// Colors
			vertex0.outMeasure = input[1].inMeasure;
			vertex1.outMeasure = input[2].inMeasure;

			// Tangent
			vertex0.outTangent = tangent;
			vertex1.outTangent = tangent;

			// World Position
			vertex0.outLightDir = viewDir;
			vertex1.outLightDir = viewDir;

			output.Append(vertex0);
			output.Append(vertex1);

		}

		output.RestartStrip();
	}		
}



