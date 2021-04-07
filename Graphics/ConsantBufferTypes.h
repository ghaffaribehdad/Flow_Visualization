#pragma once
#include <DirectXMath.h>
#include "../Options/ImGuiOptions.h"



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
	int projection;
	DirectX::XMFLOAT3 gridDiameter;
	bool periodicity;
	float particlePlanePos;
	unsigned int transparencyMode;
	unsigned int timDim;
	float streakPos;
	unsigned int currentTime;
	bool usingThreshold;
	float threshold;
};


struct BoxTube_geometryShader
{
	DirectX::XMMATRIX View;
	DirectX::XMMATRIX Proj;
	DirectX::XMFLOAT3 viewDir;
	float tubeRadius;
	DirectX::XMFLOAT3 eyePos;
	float size;
};

struct CB_pixelShader
{

	DirectX::XMFLOAT4 minColor;
	DirectX::XMFLOAT4 maxColor;
	float minMeasure;
	float maxMeasure;
	bool condition;
};

struct CB_pixelShader_Sampler
{

	float transparency;
};