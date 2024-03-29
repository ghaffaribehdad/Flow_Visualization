cbuffer PS_CBuffer
{

	float4 minColor;

};

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

	float4 rgb = minColor;

	rgb.x = rgb.x;
	rgb.y = rgb.y;
	rgb.z = rgb.z;
	rgb.w = 1.0f;

	return rgb;
}