#pragma once
#include <DirectXMath.h>

struct Vertex
{
	Vertex(){}
	Vertex(float x, float y, float z, float tx, float ty,float tz, unsigned int id, float _measure,float nx, float ny, float nz)
		:pos(x, y, z), tangent(tx,ty,tz),LineID(id),measure(_measure),normal(nx,ny,nz){}
	DirectX::XMFLOAT3 pos = { 0,0,0 };
	DirectX::XMFLOAT3 tangent = { 0,0,0 };
	unsigned int LineID = 0;
	float measure = 0;
	DirectX::XMFLOAT3 normal = { 0,0,0 };
};

struct TexCoordVertex
{
	TexCoordVertex() {}
	TexCoordVertex(float x, float y, float z, float tx, float ty)
		:pos(x,y,z),TexCoord(tx, ty){}
	DirectX::XMFLOAT3 pos = { 0,0,0 };
	DirectX::XMFLOAT2 TexCoord = { 0,0};
};






