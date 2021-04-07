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
	float3 outCenter: CENTER;
	float3 outViewPos: POS;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
	float radius : RADIUS;
	float transparency : TRANSPARENCY;
};



float4 main(PS_INPUT input) : SV_TARGET
{
	float measure = input.outMeasure - minMeasure;

	float radius = abs(distance(input.outCenter,input.outViewPos));
	if (radius < input.radius / 2)
	{
		float Projection = maxMeasure - minMeasure == 0 ? saturate(measure / (maxMeasure - minMeasure + .00001f)) : saturate(measure / (maxMeasure - minMeasure));

		float4 rgb = ((1.0 - Projection) * minColor) + (Projection * maxColor);

		float diffuse = max(dot(normalize(input.outNormal), input.outLightDir),0);
		rgb = rgb * diffuse * (1- 0.5 * radius / input.radius ) * (1 - 0.5 * radius / input.radius);


		if (condition)
			rgb.w = input.transparency;
		else
			rgb.w = 1;

		return rgb;
	}
	else
	{
		discard;
		return 0;
	}


}