cbuffer mycBuffer 
{
	float4x4 mat;
};


struct VS_INPUT
{
    float3 inPos : POSITION;
	float3 inTangent: TANGENT;
	unsigned int inLineID : LINEID;
	float inMeasure : MEASURE;
};


struct VS_OUTPUT
{
    float3 outPosition : POSITION;
	float3 outTangent: TANGENT;
	unsigned int outLineID : LINEID;
	float outMeasure : MEASURE;
};




VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
	output.outPosition = input.inPos;
	output.outTangent = input.inTangent;
	output.outLineID = input.inLineID;
	output.outMeasure = input.inMeasure;

    return output;
}
