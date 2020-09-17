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
void main(point GS_INPUT input[1], inout PointStream<GS_OUTPUT> output)
{
	GS_OUTPUT vertex0;
	vertex0.outPosition = mul(View, float4(input[0].inPosition, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outTangent = input[0].inTangent;
	vertex0.outLightDir = viewDir;
	vertex0.outNormal = float3(1, 1, 1);
	vertex0.outMeasure = input[0].inMeasure;
	output.Append(vertex0);
	output.RestartStrip();
}



