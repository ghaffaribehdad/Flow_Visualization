#include "Engine.h"

bool Engine::Initialize(HINSTANCE hInstance, std::string window_title, std::string window_class, int width, int height)
{
	// Start the timer
	this->timer.Start();

	if (!this->render_window.Initialize(this, hInstance, window_title, window_class, width, height))
	{
		return false;
	}
	if (!gfx.Initialize(this->render_window.GetHWND(),width,height))
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
				this->gfx.camera.AdjustRotation((float)me.GetPosY() * 0.005f, (float)me.GetPosX() *  0.005f, 0);
			}
		}
	}

	const float cameraSpeed = 0.001f;



	if (keyboard.KeyIsPressed('W'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetForwardVector() * cameraSpeed * dt);
	}
	if (keyboard.KeyIsPressed('S'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetBackwardVector() * cameraSpeed* dt);
	}
	if (keyboard.KeyIsPressed('A'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetLeftVector() * cameraSpeed* dt);
	}
	if (keyboard.KeyIsPressed('D'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetRightVector() * cameraSpeed* dt);
	}
	if (keyboard.KeyIsPressed(VK_SPACE))
	{
		this->gfx.camera.AdjustPosition(0.0f, cameraSpeed * dt, 0.0f);
	}
	if (keyboard.KeyIsPressed('Z'))
	{
		this->gfx.camera.AdjustPosition(0.0f, -cameraSpeed * dt, 0.0f);
	}

	// Resize the window
	if (this->resize)
	{
		OutputDebugStringA("It is resized!\n");
		this->gfx.Resize(this->render_window.GetHWND());
		this-> resize = false;
	}

	// Streamline Solver
	if (this->gfx.solverOptions.begin)
	{
		this->streamlineSolver.Initialize( //initilize cuda solver
			SEED_RANDOM,
			EULER_METHOD,
			Linear,
			this->gfx.solverOptions,
			this->gfx.GetAdapter()
		);
		this->streamlineSolver.solve();
		
		this->gfx.setCudaVertex(this->streamlineSolver.getVortices());
		this->gfx.draw_streamlines = true;

		this->gfx.solverOptions.begin = false;
	}
}

void Engine::RenderFrame()
{
	gfx.RenderFrame();
}





