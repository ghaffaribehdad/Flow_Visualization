

struct PS_INPUT
{

	float4 outPosition : SV_POSITION;
	float3 outTangent: TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
};

//struct PS_INPUT
//{
//
//	float4 outPosition : SV_POSITION;
//	float3 outTangent: TANGENT;
//	unsigned int outWorldPosition: LINEID;
//	float outMeasure : MEASURE;
//};

float4 main(PS_INPUT input) : SV_TARGET
{
	// TO-DO: Change color base on a measure
	//float4 outPosition = input.position;
	//float3 outTangent = input.tangent;
	//float4 outVelocity = input.color;

	//float3 pixelColor = {input.color.x,input.color.y,input.color.z,input.color.w};
	//float3 pixelColor = {input.color.x,0,0};
	//float4 rgb = float4(input.outMeasure,0.5f,0.5f,1);
	float4 rgb = float4(0.5f, 0.5f, 0.5f, 1);

	float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0.0f);
	rgb = rgb * diffuse;
	
	return rgb;
}