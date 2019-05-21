struct PS_INPUT
{
    float4 inPosition : SV_POSITION;
    float2 inVelocity : VELOCITY;
};

//Texture2D objTexture : TEXTURE : register(t0);
//SamplerState objSamplerState : SAMPLER : register(s0);

float4 main(PS_INPUT input) : SV_TARGET
{
	/*float3 pixelColor = objTexture.Sample(objSamplerState, input.inTexCoord);*/
	float3 pixelColor = {input.inVelocity.x,1,1};
    return float4(pixelColor, 1.0f); 
}