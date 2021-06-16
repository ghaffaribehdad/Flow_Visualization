#include "Engine.h"

bool Engine::Initialize(HINSTANCE hInstance, std::string window_title, std::string window_class, int width, int height)
{
	// Start the timer
	this->timer.Start();

	if (!this->render_window.Initialize(this, hInstance, window_title, window_class, width, height))
	{
		return false;
	}
	if (!gfx.Initialize(this->render_window.GetHWND(), width, height))
	{
		return false;
	}
	OutputDebugStringA("First Initialization!\n");
	return true;

}


bool Engine::ProcessMessages()
{
	return this->render_window.ProcessMessages();
}

void Engine::Update()
{
	// calculate the time from initialization to update
	float dt = static_cast<float>(timer.GetMilisecondsElapsed());
	timer.Restart();

	while (!keyboard.CharBufferIsEmpty())
	{
		unsigned char ch = keyboard.ReadChar();

	}

	while (!keyboard.KeyBufferIsEmpty())
	{
		KeyboardEvent kbe = keyboard.ReadKey();
		unsigned char keycode = kbe.GetKeyCode();
	}
	while (!mouse.EventBufferIsEmpty())
	{
		MouseEvent me = mouse.ReadEvent();
		if (mouse.IsRightDown())
		{
			if (me.GetType() == MouseEvent::EventType::RAW_MOVE)
			{
				this->gfx.camera.AdjustRotation((float)me.GetPosY() * gfx.mouseSpeed, (float)me.GetPosX() * gfx.mouseSpeed, 0);
				this->gfx.viewChanged();

			}
	
		}

	}

	const float cameraSpeed = 0.001f;



	if (keyboard.KeyIsPressed('W'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetViewXMVector() * 1.5f*cameraSpeed * dt);
		this->gfx.viewChanged();
	}

	if (keyboard.KeyIsPressed('U'))
	{
		this->gfx.camera.AdjustRotation((float)1 * 0.005f, 0, 0);
		this->gfx.viewChanged();
	}
	if (keyboard.KeyIsPressed('J'))
	{
		this->gfx.camera.AdjustRotation((float)-1 * 0.005f, 0, 0);
		this->gfx.viewChanged();
	}

	if (keyboard.KeyIsPressed('K'))
	{
		this->gfx.camera.AdjustRotation(0,(float)1 * 0.005f, 0);
		this->gfx.viewChanged();
	}

	if (keyboard.KeyIsPressed('H'))
	{
		this->gfx.camera.AdjustRotation(0, (float)-1 * 0.005f, 0);
		this->gfx.viewChanged();
	}

	if (keyboard.KeyIsPressed('S'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetViewXMVector() * -1.5f * cameraSpeed * dt);
		this->gfx.viewChanged();

	}
	if (keyboard.KeyIsPressed('A'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetLeftVector() * cameraSpeed * dt);
		this->gfx.viewChanged();

	}

	if (keyboard.KeyIsPressed((char)191))
	{
		this->gfx.renderImGuiOptions.hideOptions = !this->gfx.renderImGuiOptions.hideOptions;

	}

	if (keyboard.KeyIsPressed('D'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetRightVector() * cameraSpeed * dt);
		this->gfx.viewChanged();

	}
	if (keyboard.KeyIsPressed(VK_SPACE))
	{
		this->gfx.camera.AdjustPosition(gfx.camera.GetUpXMVector()*  cameraSpeed * dt);
		this->gfx.viewChanged();

	}
	if (keyboard.KeyIsPressed('Z'))
	{
		this->gfx.camera.AdjustPosition(gfx.camera.GetUpXMVector() * -cameraSpeed * dt);
		this->gfx.viewChanged();

	}

	// Resize the window
	if (this->resize)
	{
		OutputDebugStringA("It is resized!\n");
		this->gfx.Resize(this->render_window.GetHWND());
		this->gfx.renderImGuiOptions.updateStreamlines = true;
		this->gfx.raycastingOptions.resize = true;
		this->gfx.spaceTimeOptions.resize = true;
		this->resize = false;
	}
}

void Engine::RenderFrame()
{
	gfx.RenderFrame();
}

void Engine::release()
{

}