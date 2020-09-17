cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
	bool isRaycasting;
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
	float measure = input.outMeasure - minMeasure;

	float Projection = maxMeasure - minMeasure == 0 ? saturate(measure / (maxMeasure - minMeasure + .00001f)) : saturate(measure / (maxMeasure - minMeasure));

	float4 rgb = ((1.0 - Projection) * minColor) + (Projection * maxColor);

	float diffuse = max(dot(normalize(input.outNormal), input.outLightDir),0);
	rgb = rgb * diffuse;
	//rgb.w = Projection;
	rgb.w = 1;


	return rgb;
}