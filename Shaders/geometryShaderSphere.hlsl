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
	GS_OUTPUT vertex0;

	float3 tangent0 = input[0].inTangent;
	float3 inPlaneVec = normalize(cross(viewDir, tangent0));
	float3 inPlaneVecPer = normalize(cross(inPlaneVec, viewDir));



	vertex0.outCenter = input[0].inPosition;

	vertex0.outLightDir = viewDir;
	vertex0.outNormal = viewDir;
	vertex0.outMeasure = input[0].inMeasure;

	vertex0.radius = tubeRadius;

	vertex0.outPosition = mul(View, float4(input[0].inPosition + tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = input[0].inPosition + tubeRadius * inPlaneVec;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(input[0].inPosition + tubeRadius * inPlaneVecPer, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = input[0].inPosition + tubeRadius * inPlaneVecPer;
	output.Append(vertex0);

	vertex0.outPosition = mul(View, float4(input[0].inPosition - tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = input[0].inPosition - tubeRadius * inPlaneVec;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(input[0].inPosition - tubeRadius * inPlaneVecPer, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = input[0].inPosition - tubeRadius * inPlaneVecPer;
	output.Append(vertex0);


	vertex0.outPosition = mul(View, float4(input[0].inPosition + tubeRadius * inPlaneVec, 1.0f));
	vertex0.outPosition = mul(Proj, vertex0.outPosition);
	vertex0.outViewPos = input[0].inPosition + tubeRadius * inPlaneVec;
	output.Append(vertex0);


	
	
	output.RestartStrip();
}



