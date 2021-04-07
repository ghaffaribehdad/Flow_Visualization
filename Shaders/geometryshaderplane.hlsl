
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
void main(triangle GS_INPUT input[3], inout TriangleStream<GS_OUTPUT> output)
{

	for (int i = 0; i < 3; i++)
	{
		GS_OUTPUT vertex0;
		float3 position0 = input[i].inPosition;
		vertex0.outPosition = mul(View, float4(position0, 1.0f));
		vertex0.outPosition = mul(Proj, vertex0.outPosition);
		vertex0.outTangent = input[i].inTangent;
		vertex0.outLightDir = input[i].inTangent;
		vertex0.outNormal = input[i].inTangent;
		vertex0.outMeasure = input[i].inMeasure;
		output.Append(vertex0);
	}



	output.RestartStrip();
}



