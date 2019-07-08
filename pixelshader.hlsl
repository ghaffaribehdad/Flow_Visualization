struct PS_INPUT
{
    float4 inPosition : SV_POSITION;
    float2 inVelocity : VELOCITY;
	float3 inTangent: TANGENT;
};

//Texture2D objTexture : TEXTURE : register(t0);
//SamplerState objSamplerState : SAMPLER : register(s0);

float4 main(PS_INPUT input) : SV_TARGET
{
	// TO-DO: Change color base on a measure
	/*float3 pixelColor = objTexture.Sample(objSamplerState, input.inTexCoord);*/
	float3 pixelColor = {input.inVelocity.y,input.inVelocity.y,input.inVelocity.y};
    return float4(pixelColor, 1.0f); 
}