cbuffer mycBuffer
{
	float4x4 View;
	float4x4 Proj;
};

struct VS_INPUT
{
	float3 inPos : POSITION;
	float2 inTexCoord : TEXCOORD;

};


struct VS_OUTPUT
{
	float4 outPosition : SV_POSITION;
	float2 outTexCoord : TEXCOORD;
};




VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;


	float4 pos = mul(View, float4(input.inPos.xyz, 1.0f));
	pos= mul(Proj, pos);

	//output.outPosition = float4(input.inPos,1);
	output.outPosition = pos;
	output.outTexCoord = input.inTexCoord;

	return output;
}
