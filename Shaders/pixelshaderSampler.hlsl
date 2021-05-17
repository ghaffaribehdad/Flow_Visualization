
cbuffer PS_CBuffer
{
	float transparency;
};



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

Texture2D objTexture : TEXTURE;
SamplerState objSamplerState: SAMPLER;

PS_OUT main(PS_INPUT input)
{

	PS_OUT output;

	float4 color = objTexture.Sample(objSamplerState, input.inTexCoord);


	
	output.color = float4(color.xyz, 1);
	//output.color = float4(1,0,1, 1);
	output.depth = input.inPos.z;


	return output;
}