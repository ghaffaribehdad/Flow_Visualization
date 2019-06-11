#pragma once
#include <DirectXMath.h>

struct Vertex
{
	Vertex(){}
	Vertex(float x, float y, float z, float u, float v,float tx, float ty, float tz)
		:pos(x, y, z), colorID(u,v), tangent(tx,ty,tz){}
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT2 colorID;
	DirectX::XMFLOAT3 tangent;
};