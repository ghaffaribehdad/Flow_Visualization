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


	// Assume 2 light sources
	float3 light1 = normalize(float3( 1.0, 0.5, 1 ));
	float3 light2 = normalize(float3( -1, 0.5, -1 ));
	//float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0.0f);
	float diffuse = max(dot(normalize(input.outNormal), light1), 0.0f);
	diffuse += max(dot(normalize(input.outNormal), light2), 0.0f);
	rgb.x = rgb.x * diffuse;
	rgb.y = rgb.y * diffuse;
	rgb.z = rgb.z * diffuse;

	
	return rgb;
}