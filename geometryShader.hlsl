
struct VS_OUTPUT
{
	float4 position : POSITION;
	float3 tangent : TANGENT;
	int lineID : LINEID;
	float4 color : COLOR;

};

struct GS_OUTPUT
{

	float4 position : SV_POSITION;
	float3 worldPosition : WORLDPOSITION;
	float3 normal : NORMAL0;
	float3 tangent : TANGENT0;
	float4 color : COLOR;

};


[maxvertexcount(18)]
//lineadj
void main(VS_OUTPUT input[2], inout TriangleStream<GS_OUTPUT> output) {

	float StripWidth = input[0].tubeRadius * input[0].size;




	if (input[0].lineID == input[1].lineID)
	{

		float3 viewDirection = normalize(input[0].position.xyz - input[0].CameraPosition);


		float3 viewTangent0 = mul(float4(input[0].tangent, 0), View).xyz;
		float3 viewTangent1 = mul(float4(input[1].tangent, 0), View).xyz;




		float3 viewDirectionWorld = normalize(input[0].position -
			CameraPosition);
		float3 random_dir = viewDirectionWorld;

		float3 normal0 = normalize(cross(input[0].tangent, random_dir));
		float3 normal1 = normalize(cross(input[1].tangent, random_dir));



		float3 orient0 = normalize(cross(normal0, input[0].tangent));
		float3 orient1 = normalize(cross(normal1, input[1].tangent));


		float averageLength = length(input[0].position.xyz -
			input[1].position.xyz);


		float t = 360.0 / 8.0;

		GS_OUTPUT vertex0;
		GS_OUTPUT vertex1;

		for (int i = 0; i <= 8; i++)
		{

			float angle = t * i;
			float rad = (3.1415f / 180.0f) * (angle);
			float cosine = cos(rad) * StripWidth;
			float sine = sin(rad) * StripWidth;

			float3 position = input[0].position.xyz + cosine * normal0
				+ sine * orient0;
			float3 vertexNormal = normalize(position -
				input[0].position.xyz);

			vertex0.worldPosition = position;
			float3 VRCPosition = mul(float4(position, 1), View).xyz;
			vertex0.position = mul(float4(VRCPosition, 1), Proj);

			vertex0.normal = vertexNormal;
			vertex0.tangent = input[0].tangent;
			vertex0.color = input[0].color;



			position = input[1].position.xyz + cosine * normal1 + sine
				* orient1;
			vertexNormal = normalize(position - input[1].position.xyz);

			vertex1.worldPosition = position;
			VRCPosition = mul(float4(position, 1), View).xyz;
			vertex1.position = mul(float4(VRCPosition, 1), Proj);

			vertex1.normal = vertexNormal;
			vertex1.tangent = input[1].tangent;
			vertex1.color = input[1].color;



			output.Append(vertex0);
			output.Append(vertex1);


		}
		output.RestartStrip();


	}

}


