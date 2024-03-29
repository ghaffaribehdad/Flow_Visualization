#pragma once
#include <DirectXMath.h>
using namespace DirectX;


// A structure to keep camera properties
struct Camera_Prop
{
	float FOV; // Field of View
	float nearField;
	float farField;
	float eyePosition[3]; // Eye or Camera Position

	Camera_Prop(float _FOV, float _nearField, float _farField, const float* _eyePosition) :
		FOV(_FOV), nearField(_nearField), farField(_farField)
	{
		eyePosition[0] = _eyePosition[0];
		eyePosition[1] = _eyePosition[1];
		eyePosition[2] = _eyePosition[2];
	}
};


class Camera
{
public:
	Camera();
	void SetProjectionValues(float fovDegrees, float aspectRatio, float nearZ, float farZ);
	void SetParallelProjectionValues(const float & aspectRatio, const float &  viewHeight, float nearZ, float farZ);

	const XMMATRIX & GetViewMatrix() const;
	const XMMATRIX & GetProjectionMatrix() const;
	const XMMATRIX & GetParallelProjectionMatrix() const;

	const XMVECTOR & GetPositionVector() const;
	const XMFLOAT3 & GetPositionFloat3() const;
	const XMVECTOR & GetRotationVector() const;
	const XMFLOAT3 & GetRotationFloat3() const;

	void SetPosition(const XMVECTOR & pos);
	void SetPosition(float x, float y, float z);
	void SetPosition(const float * eyePos);
	void AdjustPosition(const XMVECTOR & pos);
	void AdjustPosition(float x, float y, float z);
	void SetRotation(const XMVECTOR & rot);
	void SetRotation(float x, float y, float z);
	void AdjustRotation(const XMVECTOR & rot);
	void AdjustRotation(float x, float y, float z);
	void SetLookAtPos(XMFLOAT3 lookAtPos);

	const XMVECTOR & GetForwardVector();
	const XMVECTOR & GetRightVector();
	const XMVECTOR & GetBackwardVector();
	const XMVECTOR & GetLeftVector();

	const XMFLOAT3& GetUpVector();
	const XMVECTOR& GetUpXMVector();
	const XMFLOAT3& GetViewVector();
	const XMVECTOR& GetViewXMVector();
	
	const XMMATRIX 	GetViewMatrix(XMFLOAT3 _eyePos)
	{

		XMFLOAT3 eyePos = _eyePos;
		XMVECTOR posVector = XMLoadFloat3(&eyePos);
		//XMMATRIX camRotationMatrix = XMMatrixRotationRollPitchYaw(this->rot.x, this->rot.y, this->rot.z);
		//XMMATRIX camRotationMatrix = XMMatrixRotationRollPitchYaw(0, 0,0);
		XMVECTOR camTarget = this->DEFAULT_RIGHT_VECTOR;
		//Adjust cam target to be offset by the camera's current position
		camTarget += posVector;
		//Calculate up direction based on current rotation
		XMVECTOR upDir = this->DEFAULT_UP_VECTOR;
		//Rebuild view matrix
		return XMMatrixLookAtLH(posVector, camTarget, upDir);
	}


	
private:
	void UpdateViewMatrix();
	XMVECTOR posVector;
	XMVECTOR rotVector;
	XMFLOAT3 pos;
	XMFLOAT3 rot;
	XMMATRIX viewMatrix;
	XMMATRIX projectionMatrix;
	XMMATRIX parallelProjectionMatrix;
	XMFLOAT3 viewDir;
	XMFLOAT3 upDir;
	XMVECTOR v_viewDir;
	XMVECTOR v_upDir;


	const XMVECTOR DEFAULT_FORWARD_VECTOR = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
	const XMVECTOR DEFAULT_UP_VECTOR = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	const XMVECTOR DEFAULT_BACKWARD_VECTOR = XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f);
	const XMVECTOR DEFAULT_LEFT_VECTOR = XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f);
	const XMVECTOR DEFAULT_RIGHT_VECTOR = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);

	XMVECTOR vec_forward;
	XMVECTOR vec_left;
	XMVECTOR vec_right;
	XMVECTOR vec_backward;
};