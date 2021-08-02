
static int layerLimit = 50;
static uint SortedFragmentsColor[50 + 1];
static uint SortedFragmentsCoverage[50 + 1];
static float SortedFragmentsDepth[50 + 1];


struct Fragment_And_Link_Buffer_STRUCT
{
	uint    Color;
	uint	Coverage;       // Coverage is only used in the MSAA case
	float	Depth;
	uint    uNext;
};

Buffer<uint>  StartOffsetBuffer  										:register(t0);
StructuredBuffer<Fragment_And_Link_Buffer_STRUCT> FragmentAndLinkBufferSRV	:register(t1);

cbuffer PS_CBuffer
{

	int viewportWidth;
	int viewportHeight;

};

uint UnpackCoverageIntoUint(uint uDepthAndCoverage)
{
	return (uDepthAndCoverage & 0xff);
}

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
	Fragment_And_Link_Buffer_STRUCT Element;
	int counter = 0;

	while (uOffset != 0xFFFFFFFF && counter != layerLimit)
	{
		counter++;
		// Retrieve fragment at current offset
		Element = FragmentAndLinkBufferSRV[uOffset];

		// Copy fragment color and depth into sorted list
		// (float depth is directly cast into a uint - this is OK since 
		// depth comparisons will still work after casting)
		SortedFragmentsColor[nNumFragments] = Element.Color;
		SortedFragmentsCoverage[nNumFragments] = Element.Coverage;
		SortedFragmentsDepth[nNumFragments] = Element.Depth;

		// Sort fragments in front to back (increasing) order using insertion sorting
		// max(j-1,0) is used to cater for the case where nNumFragments=0
		int j = nNumFragments;
		[loop]while ((j > 0) && (SortedFragmentsDepth[max(j - 1, 0)] > SortedFragmentsDepth[j]))
		{
			// Swap required
			int jminusone = max(j - 1, 0);
			uint TmpColor = SortedFragmentsColor[j];
			uint TmpCoverage = SortedFragmentsCoverage[j];
			float TmpDepth = SortedFragmentsDepth[j];

			SortedFragmentsColor[j] = SortedFragmentsColor[jminusone];
			SortedFragmentsCoverage[j] = SortedFragmentsCoverage[jminusone];
			SortedFragmentsDepth[j] = SortedFragmentsDepth[jminusone];


			SortedFragmentsColor[jminusone] = TmpColor;
			SortedFragmentsCoverage[jminusone] = TmpCoverage;
			SortedFragmentsDepth[jminusone] = TmpDepth;
			j--;
		}

		// Increase number of fragment
		nNumFragments = min(nNumFragments + 1, 100);

		// Retrieve next offset
		uOffset = Element.uNext;
	}


	//float3 vCurrentColorSample[4] = { float3(1,1,1),float3(1,1,1) ,float3(1,1,1) ,float3(1,1,1) };
	float4 vCurrentColor = float4(1, 1, 1,0);

	//// Render fragments using SRCALPHA-INVSRCALPHA blending
	//for (int k = nNumFragments - 1; k >= 0; k--)
	//{
	//	{
	//		float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragmentsColor[k]);
	//		uint uCoverage = SortedFragmentsCoverage[k];

	//		for (uint uSample = 0; uSample < 4; uSample++)
	//		{
	//			float fIsSampleCovered = (uCoverage & (1 << uSample)) ? 1.0 : 0.0;
	//			vCurrentColorSample[uSample].xyz = lerp(vCurrentColor.xyz, vFragmentColor.xyz, vFragmentColor.w * fIsSampleCovered);
	//		}
	//}

	for (int k = nNumFragments - 1; k >= 0; k--)
	{
		float4 vFragmentColor = UnpackUintIntoFloat4(SortedFragmentsColor[k]);
		vCurrentColor.xyz = lerp(vCurrentColor.xyz, vFragmentColor.xyz, vFragmentColor.w);
	}


	//// Resolve samples into a single color
	//float4 vCurrentColor = float4(0, 0, 0, 0);
	//[unroll]for (uint uSample = 0; uSample < 4; uSample++)
	//{
	//	vCurrentColor.xyz += vCurrentColorSample[uSample];
	//}
	//vCurrentColor.xyz *= (1.0 / 4);
	

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
