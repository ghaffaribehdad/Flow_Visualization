#pragma once
#include <DirectXMath.h>

struct CB_VS_vertexshader
{
	DirectX::XMMATRIX mat;
	float color = 0.5f;
};
// 8 Bytes
// + 4 Bytes



struct Tube_geometryShader
{
	DirectX::XMFLOAT3 viewDir;
	float tubeRadius;
	float size;
	DirectX::XMFLOAT3 cameraPosition;
};
