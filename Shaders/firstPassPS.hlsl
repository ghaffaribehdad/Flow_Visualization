
#define MAX_24BIT_UINT  ( (1<<24) - 1 )

struct Fragment_And_Link_Buffer_STRUCT
{
	uint    uPixelColor;
	float    uDepthAndCoverage;       // Coverage is only used in the MSAA case
	uint    uNext;
};

RWByteAddressBuffer StartOffsetBuffer  										:register(u1);
RWStructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBuffer	:register(u2);




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



float4 main(PS_INPUT input) : SV_TARGET
{

	//else the color gradient
	float measure = abs(input.outMeasure);
	float4 rgb_compl_max = float4(1.0f, 1.0f, 1.0, 1.0f) - maxColor;
	float Projection = maxMeasure == 0 ? saturate(measure / (maxMeasure + .00001f)) : saturate(measure / maxMeasure);
	float4 rgb = { 0.0f,0.0f,0.0f,0.0f };

	rgb = (Projection)* rgb_compl_max + maxColor;
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
	rgb.w = Projection;


	


	uint uPixelCount = FragmentAndLinkBuffer.IncrementCounter(); // store the current counter value and increase it by 1
	uint linearindex =   4*(input.outPosition.y * viewportWidth + input.outPosition.x);
	uint uOldStartOffset;
	StartOffsetBuffer.InterlockedExchange(linearindex, uPixelCount, uOldStartOffset);

	Fragment_And_Link_Buffer_STRUCT Element;
	Element.uPixelColor = (((uint)(rgb.x * 255)) << 24) | (((uint)(rgb.y * 255)) << 16) | (((uint)(rgb.z * 255)) << 8) | (uint)(rgb.w * 255);
	//Element.uDepthAndCoverage = input.outPosition.z * MAX_24BIT_UINT));
	Element.uDepthAndCoverage = input.outPosition.z;
	Element.uNext = uOldStartOffset;
	FragmentAndLinkBuffer[uPixelCount] = Element;


	// This won't write anything into the RT because color writes are off    
	return float4(0,0,0,0);

}