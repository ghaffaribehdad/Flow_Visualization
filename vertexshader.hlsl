cbuffer mycBuffer : register(b0)
{
    float4x4 mat;
	float color;
};


struct VS_INPUT
{
    float3 inPos : POSITION;
	float3 inTangent: TANGENT;
	float inLineID : LINEID;
	float4 inColor : COLOR;
};


struct VS_OUTPUT
{
    float4 outPosition : SV_POSITION;
	float3 outTangent: TANGENT;
	float outLineID : LINEID;
	float4 outColor : COLOR;
	float color : COLORCONSTANT;

};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
    output.outPosition = mul(float4(input.inPos, 1.0f), mat);
	output.outTangent = input.inTangent;
	output.outLineID = input.inLineID;
	output.outColor = input.inColor;
	output.color = color;

    return output;
}

 //Position must be exatly the same with the layout description semantic
// SV_POSITION is HLSL syntax and gives the return type