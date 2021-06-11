
#define MAX_24BIT_UINT  ( (1<<24) - 1 )

struct Fragment_And_Link_Buffer_STRUCT
{
	uint    uPixelColor;
	float   uDepthAndCoverage;       // Coverage is only used in the MSAA case
	uint	coverage;
	uint    uNext;
};

RWByteAddressBuffer StartOffsetBuffer  										:register(u1);
RWStructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBuffer	:register(u2);




float4 colorcoding(float3 rgb_min, float3 rgb_max, float value, float min_val, float max_val)
{

	float3 rgb = { 0,0,0 };
	float y_saturated = 0.0f;


	float min = 0;
	float max = max_val - min_val;
	float val = value - min_val;

	float sat = saturate(value / (max - min));

	rgb = (1 - sat) * rgb_min + sat * rgb_max;

	//if (value < 0)
	//{
	//	float3 rgb_min_complement = float3(1, 1, 1) - rgb_min;
	//	y_saturated = saturate(abs(value / min_val));
	//	rgb = rgb_min_complement * (1 - y_saturated) + rgb_min;
	//}
	//else
	//{
	//	float3 rgb_max_complement = float3(1, 1, 1) - rgb_max;
	//	y_saturated = saturate(max_val / value);
	//	//rgb = rgb_max_complement * abs(y_saturated - 1) + rgb_max;
	//	rgb = rgb_max_complement * (1 - y_saturated) + rgb_max;
	//}





	return float4(rgb.xyz, 1);
}


cbuffer PS_CBuffer
{

	float4 minColor;
	float4 maxColor;
	float minMeasure;
	float maxMeasure;
	int viewportWidth;
	int viewportHeight;
	bool condition; // In this case transparency
	
};


struct PS_INPUT
{

	float4 outPosition : SV_POSITION;
	uint  uCoverage : SV_COVERAGE;
	float3 outTangent: TANGENT;
	float3 outLightDir: LIGHTDIR;
	float3 outNormal : NORMAL;
	float outMeasure : MEASURE;
	float transparency : TRANSPARENCY;
};



float4 main(PS_INPUT input) : SV_TARGET
{

	//else the color gradient
	//float measure = abs(input.outMeasure);
	float measure = input.outMeasure;
	//float4 rgb_compl_max = float4(1.0f, 1.0f, 1.0, 1.0f) - maxColor;
	float4 rgb = { 0.0f,0.0f,0.0f,0.0f };

	// calculate color coding
	if (condition)
	{

		rgb = float4(maxColor.xyz, 1);
	}
	else
	{
		rgb = colorcoding(minColor.xyz, maxColor.xyz, measure, minMeasure, maxMeasure); // color coding
	}

	
	// Compute shading
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


	if (condition)
	{

		rgb.w = 1 - pow((maxMeasure - minMeasure == 0 ? saturate((measure - minMeasure) / (maxMeasure - minMeasure + 0.001)) : saturate((measure - minMeasure) / (maxMeasure - minMeasure))),1);
	}
	else
	{
		rgb.w = 1;
	}



	// OIT

	uint uPixelCount = FragmentAndLinkBuffer.IncrementCounter(); // store the current counter value and increase it by 1
	uint linearindex =   4*(input.outPosition.y * viewportWidth + input.outPosition.x);
	uint uOldStartOffset;
	StartOffsetBuffer.InterlockedExchange(linearindex, uPixelCount, uOldStartOffset);

	Fragment_And_Link_Buffer_STRUCT Element;
	Element.uPixelColor = (((uint)(rgb.x * 255)) << 24) | (((uint)(rgb.y * 255)) << 16) | (((uint)(rgb.z * 255)) << 8) | (uint)(rgb.w * 255);
	//Element.uDepthAndCoverage = input.outPosition.z * MAX_24BIT_UINT));
	Element.uDepthAndCoverage = input.outPosition.z;
	Element.uNext = uOldStartOffset;
	Element.coverage = input.uCoverage;
	FragmentAndLinkBuffer[uPixelCount] = Element;


	// This won't write anything into the RT because color writes are off    
	return float4(0,0,0,0);

}