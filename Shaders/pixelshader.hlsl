cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
	int viewportWidth;
	int viewportHeight;
	bool saturation;
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

	// If saturation is needed
	if (saturation)
	{
		//else the color gradient
		float4 rgb_compl_max = float4(1.0f, 1.0f, 1.0, 1.0f) - maxColor;

		float Projection = maxMeasure == 0 ? saturate(measure / (maxMeasure  + .00001f)) : saturate(measure / maxMeasure);



		rgb = (Projection )* rgb_compl_max + maxColor;

		float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0);
		float3 reflection = 2.0 * dot(input.outNormal, input.outLightDir) * input.outNormal - input.outLightDir;
		reflection = normalize(reflection);
		float cos_angle = dot(reflection, input.outLightDir);
		cos_angle = clamp(cos_angle, 0.0, 1.0);
		float u_Shininess = 0.1f;
		cos_angle = pow(cos_angle, u_Shininess);
		float4 specular = { 0.0f,0.0f,0.0f,0.0f };
		if (cos_angle > 0.0f)
		{
			float4 specular = float4(1.0, 1.0f, 1.0f, 1.0f) * cos_angle;
		}
		rgb = rgb * diffuse;
		rgb += specular;
		rgb.w = 1;

		return rgb;
	}


	rgb = colorcoding(minColor.xyz, maxColor.xyz, measure, minMeasure, maxMeasure); // color coding

	float diffuse = max(dot(normalize(input.outNormal), input.outLightDir), 0);
	float3 reflection = 2.0 * dot(input.outNormal, input.outLightDir) * input.outNormal - input.outLightDir;
	reflection = normalize(reflection);
	float cos_angle = dot(reflection, input.outLightDir);
	cos_angle = clamp(cos_angle, 0.0, 1.0);
	float u_Shininess = 0.1f;
	cos_angle = pow(cos_angle, u_Shininess);
	float4 specular = { 0.0f,0.0f,0.0f,0.0f };
	if (cos_angle > 0.0f)
	{
		float4 specular = float4(1.0, 1.0f, 1.0f, 1.0f) * cos_angle;
	}
	rgb = rgb * diffuse;
	rgb += specular;
	rgb.w = 1;
	return rgb;
}