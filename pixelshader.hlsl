struct PS_INPUT
{
    float4 inPosition : SV_POSITION;
	float3 inTangent: TANGENT;
	float inLineID : LINEID;
	float4 inColor : COLOR;
	float color : COLORCONSTANT;
};

//Texture2D objTexture : TEXTURE : register(t0);
//SamplerState objSamplerState : SAMPLER : register(s0);

float4 main(PS_INPUT input) : SV_TARGET
{
	// TO-DO: Change color base on a measure
	/*float3 pixelColor = objTexture.Sample(objSamplerState, input.inTexCoord);*/
	//float3 pixelColor = {input.inColor.x,input.inColor.x,input.inColor.x};
	float3 pixelColor = {input.color,0,0};
    return float4(pixelColor, 1.0f); 
}

