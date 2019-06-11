cbuffer mycBuffer : register(b0)
{
    float4x4 mat;
};


struct VS_INPUT
{
    float3 inPos : POSITION;
    float2 inVelocity : VELOCITY;
	float3 inTangent : TANGENT;

};


struct VS_OUTPUT
{
    float4 outPosition : SV_POSITION;
    float2 outVelocity : VELOCITY;
	float3 outTangent : TANGENT;
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
    output.outPosition = mul(float4(input.inPos, 1.0f), mat);
    output.outVelocity = input.inVelocity;
	output.outTangent = input.inTangent;
    return output;
}

 //Position must be exatly the same with the layout description semantic
// SV_POSITION is HLSL syntax and gives the return type