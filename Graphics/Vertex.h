#pragma once
#include <DirectXMath.h>

struct Vertex
{
	Vertex(){}
	Vertex(float x, float y, float z, float tx, float ty,float tz, unsigned int id, float _measure)
		:pos(x, y, z), tangent(tx,ty,tz),LineID(id),measure(_measure){}
	DirectX::XMFLOAT3 pos = { 0,0,0 };
	DirectX::XMFLOAT3 tangent = { 0,0,0 };
	unsigned int LineID = 0;
	float measure = 0;
};



