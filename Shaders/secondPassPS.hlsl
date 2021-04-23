#define MAX_24BIT_UINT  ( (1<<24) - 1 )
static uint2 SortedFragments[200 + 1];

struct Fragment_And_Link_Buffer_STRUCT
{
	uint    uPixelColor;
	uint    uDepthAndCoverage;       // Coverage is only used in the MSAA case
	uint    uNext;
};

Buffer<uint>  StartOffsetBuffer  										:register(t0);
StructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBufferSRV	:register(t1);

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


float4 UnpackUintIntoFloat4(uint uValue)
{
	return float4(((uValue & 0xFF000000) >> 24) / 255.0, ((uValue & 0x00FF0000) >> 16) / 255.0, ((uValue & 0x0000FF00) >> 8) / 255.0, ((uValue & 0x000000FF)) / 255.0);
}


struct PS_INPUT
{
	float4 inPos : SV_POSITION;
	float2 inTexCoord: TEXCOORD;
};



//struct PS_OUT
//{
//	float4 color : SV_Target;
//	float depth : SV_Depth;
//};


float4 main(PS_INPUT input) : SV_Target
{

	

	uint linearindex = uint(input.inPos.x + input.inPos.y * viewportWidth);
	uint uOffset = StartOffsetBuffer.Load(linearindex);
	uint first = uOffset;
	int nNumFragments = 0;

	while (uOffset != 0xFFFFFFFF)
	{
		Fragment_And_Link_Buffer_STRUCT Element = FragmentAndLinkBufferSRV[uOffset];
		SortedFragments[nNumFragments] = uint2(Element.uPixelColor, Element.uDepthAndCoverage);
		int j = nNumFragments;

		while ((j > 0) && (SortedFragments[max(j - 1, 0)].y > SortedFragments[j].y))
		{
			// Swap required
			int jminusone = max(j - 1, 0);
			uint2 Tmp = SortedFragments[j];
			SortedFragments[j] = SortedFragments[jminusone];
			SortedFragments[jminusone] = Tmp;
			j--;
		}

		// Increase number of fragment
		nNumFragments = min(nNumFragments + 1, 100);
		// Retrieve next offset
		uOffset = Element.uNext;


	}

	float4 vCurrentColor = { 1,1,1,1 };

	for (int k = nNumFragments - 1; k >= 0; k--)
	{
		float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragments[k].x);
		vCurrentColor.xyz = lerp(vCurrentColor.xyz, vFragmentColor.xyz, vFragmentColor.w);
	}

	return vCurrentColor;
}