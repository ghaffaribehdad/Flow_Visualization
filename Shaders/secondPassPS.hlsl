
static int layerLimit = 100;
static uint SortedFragmentsColor[100 + 1];
static float SortedFragmentsDepth[100 + 1];

struct Fragment_And_Link_Buffer_STRUCT
{
	uint    uPixelColor;
	float    uDepthAndCoverage;       // Coverage is only used in the MSAA case
	uint    uNext;
};

Buffer<uint>  StartOffsetBuffer  										:register(t0);
StructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBufferSRV	:register(t1);

cbuffer PS_CBuffer
{

	int viewportWidth;
	int viewportHeight;

};


float4 UnpackUintIntoFloat4(uint uValue)
{
	return float4(((uValue & 0xFF000000) >> 24) / 255.0, ((uValue & 0x00FF0000) >> 16) / 255.0, ((uValue & 0x0000FF00) >> 8) / 255.0, ((uValue & 0x000000FF)) / 255.0);
}

uint UnpackDepthIntoUint(uint uDepthAndCoverage)
{
	return (uint)(uDepthAndCoverage >> 8);
}

struct PS_INPUT
{
	float4 inPos : SV_POSITION;
	float2 inTexCoord: TEXCOORD;
};


struct PS_OUT
{
	float4 color : SV_Target;
	float depth : SV_Depth;
};

PS_OUT main(PS_INPUT input) 
{

	uint linearindex = uint(input.inPos.x + input.inPos.y * viewportWidth);
	uint uOffset = StartOffsetBuffer.Load(linearindex);
	uint first = uOffset;
	int nNumFragments = 0;

	int counter = 0;
	while (uOffset != 0xFFFFFFFF && counter != layerLimit)
	{
		counter++;

		Fragment_And_Link_Buffer_STRUCT Element = FragmentAndLinkBufferSRV[uOffset];
		//SortedFragments[nNumFragments] = uint2(Element.uPixelColor, Element.uDepthAndCoverage);
		SortedFragmentsColor[nNumFragments] = Element.uPixelColor;
		SortedFragmentsDepth[nNumFragments] = Element.uDepthAndCoverage;
		int j = nNumFragments;


		while ((j > 0) && (SortedFragmentsDepth[max(j - 1, 0)] > SortedFragmentsDepth[j]))
		{
			// Swap required
			int jminusone = max(j - 1, 0);
			float TmpDepth = SortedFragmentsDepth[j];
			uint TmpColor = SortedFragmentsColor[j];
			SortedFragmentsColor[j] = SortedFragmentsColor[jminusone];
			SortedFragmentsDepth[j] = SortedFragmentsDepth[jminusone];
			SortedFragmentsColor[jminusone] = TmpColor;
			SortedFragmentsDepth[jminusone] = TmpDepth;
			j--;
		}

		// Increase number of fragment
		nNumFragments = min(nNumFragments + 1, 100);
		// Retrieve next offset
		uOffset = Element.uNext;

	}




	float4 vCurrentColor = { 1,1,1,0 };

	for (int k = nNumFragments - 1; k >= 0; k--)
	{
		float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragmentsColor[k]);
		vCurrentColor.xyz = lerp(vCurrentColor.xyz, vFragmentColor.xyz, vFragmentColor.w);
	}

	PS_OUT output;
	if (nNumFragments != 0)
	{
		output.color = vCurrentColor;
		output.depth = SortedFragmentsDepth[0];
	}
	else
	{
		output.color = float4(1, 1, 1, 0);
		output.depth = 1;
	}


	return output;

}
