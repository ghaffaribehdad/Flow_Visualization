cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
	bool condition;
};



struct PS_INPUT
{

	float4 outPosition : SV_POSITION;
	float3 outTangent: TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
	float transparency : TRANSPARENCY;
};



float4 main(PS_INPUT input) : SV_TARGET
{
	if (condition)
	{
		float measure = input.outMeasure - minMeasure;

		float Projection = maxMeasure - minMeasure == 0 ? saturate(measure / (maxMeasure - minMeasure + .00001f)) : saturate(measure / (maxMeasure - minMeasure));

		float4 rgb = ((1.0 - Projection) * minColor) + (Projection * maxColor);

		float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0);
		rgb = rgb * diffuse;
		rgb.w = input.transparency;

		return rgb;
	}
	
	float measure = input.outMeasure;

	float Projection = maxMeasure - minMeasure == 0 ? saturate(measure / (maxMeasure - minMeasure + .00001f)) : saturate(measure / (maxMeasure - minMeasure));

	float4 rgb = minColor;

	float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0);
	rgb = rgb * diffuse;
	rgb.w = 1;

	return rgb;

	
}