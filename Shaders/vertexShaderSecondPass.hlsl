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


	output.outPosition = float4(input.inPos.x, input.inPos.y, input.inPos.z, 1);
	output.outTexCoord = input.inTexCoord;

	return output;
}
