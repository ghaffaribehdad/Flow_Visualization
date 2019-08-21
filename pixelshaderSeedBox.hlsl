

struct PS_INPUT
{

	float4 outPosition : SV_POSITION;
	float3 outTangent: TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
};


float4 main(PS_INPUT input) : SV_TARGET
{

	float4 rgb = float4(0.5f, 1.0f, 0.5f, 1);

	float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0.0f);
	rgb = rgb * diffuse;
	
	return rgb;
}