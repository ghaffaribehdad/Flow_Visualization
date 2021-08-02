
#define MAX_24BIT_UINT  ( (1<<24) - 1 )

struct Fragment_And_Link_Buffer_STRUCT
{
	uint    Color;
	uint	Coverage;       // Coverage is only used in the MSAA case
	float	Depth;
	uint    uNext;
};

RWByteAddressBuffer StartOffsetBuffer  										:register(u1);
RWStructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBuffer	:register(u2);



uint PackFloat4IntoUint(float4 vValue)
{
	return (((uint)(vValue.x * 255)) << 24) | (((uint)(vValue.y * 255)) << 16) | (((uint)(vValue.z * 255)) << 8) | (uint)(vValue.w * 255);
}

float4 UnpackUintIntoFloat4(uint uValue)
{
	return float4(((uValue & 0xFF000000) >> 24) / 255.0, ((uValue & 0x00FF0000) >> 16) / 255.0, ((uValue & 0x0000FF00) >> 8) / 255.0, ((uValue & 0x000000FF)) / 255.0);
}

// Pack depth into 24 MSBs
uint PackDepthIntoUint(float fDepth)
{
	return ((uint)(fDepth * MAX_24BIT_UINT)) << 8;
}

// Pack depth into 24 MSBs and coverage into 8 LSBs
uint PackDepthAndCoverageIntoUint(float fDepth, uint uCoverage)
{
	return (((uint)(fDepth * MAX_24BIT_UINT)) << 8) | uCoverage;
}

uint UnpackDepthIntoUint(uint uDepthAndCoverage)
{
	return (uint)(uDepthAndCoverage >> 8);
}

uint UnpackCoverageIntoUint(uint uDepthAndCoverage)
{
	return (uDepthAndCoverage & 0xff);
}

uint PackNormalIntoUint(float3 n)
{
	uint3 i3 = (uint3) (n * 127.0f + 127.0f);
	return i3.r + (i3.g << 8) + (i3.b << 16);
}

float3 UnpackNormalIntoFloat3(uint n)
{
	float3 n3 = float3(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff);
	return (n3 - 127.0f) / 127.0f;
}

uint PackNormalAndCoverageIntoUint(float3 n, uint uCoverage)
{
	uint3 i3 = (uint3) (n * 127.0f + 127.0f);
	return i3.r + (i3.g << 8) + (i3.b << 16) + (uCoverage << 24);
}




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
	float Ka;
	float Kd;
	float Ks;
	float shininessVal;
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

	Element.Color = PackFloat4IntoUint(rgb);
	Element.Coverage = input.uCoverage;
	Element.Depth = input.outPosition.z;
	Element.uNext = uOldStartOffset;
	//Element.coverage = input.uCoverage;
	FragmentAndLinkBuffer[uPixelCount] = Element;


	// This won't write anything into the RT because color writes are off    
	return float4(0,0,0,0);

}