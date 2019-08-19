#pragma once
#include <DirectXMath.h>

struct Vertex
{
	Vertex(){}
	Vertex(float x, float y, float z, float tx, float ty,float tz, unsigned int id, float _measure)
		:pos(x, y, z), tangent(tx,ty,tz),LineID(id),measure(_measure){}
	DirectX::XMFLOAT3 pos;
	DirectX::XMFLOAT3 tangent;
	unsigned int LineID;
	float measure;
};



