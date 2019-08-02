#pragma once
#include <DirectXMath.h>

struct Vertex
{
	Vertex(){}
	Vertex(float x, float y, float z, float tx, float ty,float tz, float id, float r, float g, float b, float a)
		:pos(x, y, z), tangent(tx,ty,tz),LineID(id),color(r,g,b,a){}
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT3 tangent;
	float LineID;
	DirectX::XMFLOAT4 color;
};



