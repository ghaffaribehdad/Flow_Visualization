#pragma once
#include <DirectXMath.h>

struct CB_VS_vertexshader
{
	DirectX::XMMATRIX mat;

};
// 8 Bytes



struct Tube_geometryShader
{
	DirectX::XMMATRIX View;
	DirectX::XMMATRIX Proj;
	DirectX::XMFLOAT3 viewDir;
	float tubeRadius;
	DirectX::XMFLOAT3 eyePos;
	float size;
};
