#include "LineRendering.h"


bool LineRendering::initializeShaders()
{

	

	if (this->rasterizerstate.Get() == nullptr)
	{
		// Create Rasterizer state
		D3D11_RASTERIZER_DESC rasterizerDesc;
		ZeroMemory(&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));

		rasterizerDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
		rasterizerDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE; // CULLING could be set to none
		rasterizerDesc.MultisampleEnable = true;
		rasterizerDesc.AntialiasedLineEnable = true;
		//rasterizerDesc.FrontCounterClockwise = TRUE;//= 1;

		HRESULT hr = this->device->CreateRasterizerState(&rasterizerDesc, this->rasterizerstate.GetAddressOf());
		if (FAILED(hr))
		{
			ErrorLogger::Log(hr, "Failed to Create rasterizer state.");
			return false;
		}
	}
	
	this->deviceContext->IASetInputLayout(this->vertexshader.GetInputLayout());		// Set the input layout
	this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP);// set the primitive topology
	this->deviceContext->RSSetState(this->rasterizerstate.Get());					// set the rasterizer state
	this->deviceContext->VSSetShader(vertexshader.GetShader(), NULL, 0);			// set vertex shader
	this->deviceContext->PSSetShader(pixelshader.GetShader(), NULL, 0);		
	// set pixel shader



	return true;
}


bool LineRendering::initializeConstantBuffer(Camera& camera)
{

	HRESULT hr = this->GS_constantBuffer.Initialize(this->device.Get(), this->deviceContext.Get());
	if (FAILED(hr))
	{
		ErrorLogger::Log(hr, "Failed to Create Constant buffer.");
		return false;
	}

	DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();

	// Set attributes of constant buffer for geometry shader
	GS_constantBuffer.data.View = world * camera.GetViewMatrix();
	GS_constantBuffer.data.Proj = camera.GetProjectionMatrix();
	GS_constantBuffer.data.eyePos = camera.GetPositionFloat3();
	GS_constantBuffer.data.tubeRadius = .1f;
	GS_constantBuffer.data.viewDir = camera.GetViewVector();

	// Update Constant Buffer
	GS_constantBuffer.ApplyChanges();

	//set the constant buffer
	this->deviceContext->GSSetConstantBuffers(0, 1, this->GS_constantBuffer.GetAddressOf());


	return true;
}