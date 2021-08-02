cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
	int viewportWidth;
	int viewportHeight;
	bool condition; // In this case transparency
	float Ka;
	float Kd;
	float Ks;
	float shininessVal;
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


float4 colorcoding(float3 rgb_min, float3 rgb_max, float value, float min_val, float max_val)
{

	float3 rgb = { 0,0,0 };
	float y_saturated = 0.0f;


	float min = 0;
	float max = max_val - min_val;
	float val = value - min_val;

	float sat = saturate(value / (max - min));

	rgb = (1 - sat) * rgb_min + sat * rgb_max;


	return float4(rgb.xyz, 1);
}




float4 main(PS_INPUT input) : SV_TARGET
{

	float4 rgb = { 0.0f,0.0f,0.0f,0.0f };
	float measure = input.outMeasure;

	// calculate color coding
	if (condition)
	{

		//rgb = float4(maxColor.xyz, 1);
		rgb = colorcoding(minColor.xyz, maxColor.xyz, measure, minMeasure, maxMeasure); // color coding
	}
	else
	{
		rgb = colorcoding(minColor.xyz, maxColor.xyz, measure, minMeasure, maxMeasure); // color coding
	}


	float3 L = normalize(input.outLightDir);
	float3 N = normalize(input.outNormal);
	float3 R = normalize(2.0 * dot(N, L) * N - L);
	float3 V = normalize(-L); // Vector to viewer
	float specAngle = max(dot(R, V), 0.0);
	float specular = pow(specAngle, shininessVal);
	float lambertian = max(dot(N, L), 0.0);


	rgb.xyz = Ka * float3(1, 1, 1) + Kd * lambertian * rgb.xyz +
		Ks * specular * float3(1, 1, 1);



	rgb.w = 1;
	return rgb;
}