cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
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



float4 main(PS_INPUT input) : SV_TARGET
{
	// If saturation is needed
	if (saturation)
	{
		//else the color gradient
		float measure = abs(input.outMeasure);
		float4 rgb_compl_max = float4(1.0f, 1.0f, 1.0, 1.0f) - maxColor;

		float Projection = maxMeasure == 0 ? saturate(measure / (maxMeasure  + .00001f)) : saturate(measure / maxMeasure);


		float4 rgb = { 0.0f,0.0f,0.0f,0.0f };

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
	//else the color gradient
	float measure = abs(input.outMeasure);
	float4 rgb_compl_min = float4(1.0f, 1.0f, 1.0, 1.0f) - minColor;
	float4 rgb_compl_max = float4(1.0f, 1.0f, 1.0, 1.0f) - maxColor;

	float Projection = maxMeasure - minMeasure == 0 ? saturate((measure - minMeasure) / (maxMeasure - minMeasure + .00001f) ): saturate((measure- minMeasure)/ (maxMeasure - minMeasure));
	if (Projection < 0)
		Projection = 0;

	float4 rgb = { 0.0f,0.0f,0.0f,0.0f };
	if(Projection < 0.5f)
	{
		rgb = (Projection * 2.0f)* rgb_compl_min + minColor;
	}
	else
	{
		rgb = (1.0f - ((Projection - 0.5f) *2.0f)) * rgb_compl_max + maxColor;
	}






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