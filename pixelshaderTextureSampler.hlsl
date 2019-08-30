




struct PS_INPUT
{
	float4 inPos : SV_POSITION;
	float2 inTexCoord: TEXCOORD;
};

Texture2D objTexture : TEXTURE;
SamplerState objSamplerState: SAMPLER; 

float4 main(PS_INPUT input) : SV_TARGET
{
	

	float4 value = objTexture.Sample(objSamplerState, input.inTexCoord);

	return value;
}