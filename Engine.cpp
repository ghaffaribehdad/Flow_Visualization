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
			this->gfx.solverOptions.userInterruption = true;
			if (me.GetType() == MouseEvent::EventType::RAW_MOVE)
			{
				this->gfx.camera.AdjustRotation((float)me.GetPosY() * 0.005f, (float)me.GetPosX() *  0.005f, 0);
			}
		}

		if (mouse.IsLeftDown())
		{
			this->gfx.solverOptions.userInterruption = true;
		}
	}

	const float cameraSpeed = 0.005f;



	if (keyboard.KeyIsPressed('W'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetForwardVector() * cameraSpeed * dt);
		this->gfx.solverOptions.userInterruption = true;
	}
	if (keyboard.KeyIsPressed('S'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetBackwardVector() * cameraSpeed* dt);
		this->gfx.solverOptions.userInterruption = true;
	}
	if (keyboard.KeyIsPressed('A'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetLeftVector() * cameraSpeed* dt);
		this->gfx.solverOptions.userInterruption = true;
	}
	if (keyboard.KeyIsPressed('D'))
	{
		this->gfx.camera.AdjustPosition(this->gfx.camera.GetRightVector() * cameraSpeed* dt);
		this->gfx.solverOptions.userInterruption = true;
	}
	if (keyboard.KeyIsPressed(VK_SPACE))
	{
		this->gfx.camera.AdjustPosition(0.0f, cameraSpeed * dt, 0.0f);
		this->gfx.solverOptions.userInterruption = true;
	}
	if (keyboard.KeyIsPressed('Z'))
	{
		this->gfx.camera.AdjustPosition(0.0f, -cameraSpeed * dt, 0.0f);
		this->gfx.solverOptions.userInterruption = true;
	}

	// Resize the window
	if (this->resize)
	{
		OutputDebugStringA("It is resized!\n");
		this->gfx.Resize(this->render_window.GetHWND());
		this-> resize = false;
	}

	// Streamline Solver
	if (this->gfx.solverOptions.beginStream)
	{
		gfx.solverOptions.p_Adapter = this->gfx.GetAdapter();
		gfx.solverOptions.p_vertexBuffer = this->gfx.GetVertexBuffer();

		if (gfx.solverOptions.precision == 32)
		{

			this->streamlineSolver_float.Initialize(this->gfx.solverOptions);
			this->streamlineSolver_float.solve();
			this->streamlineSolver_float.FinalizeCUDA();
		}
		// under construction
		else if(gfx.solverOptions.precision == 64)
		{
		}

		this->gfx.solverOptions.beginStream = false;
		this->gfx.showLines = true;
	}

	// Pathline Solver
	if (this->gfx.solverOptions.beginPath)
	{
		gfx.solverOptions.p_Adapter = this->gfx.GetAdapter();
		gfx.solverOptions.p_vertexBuffer = this->gfx.GetVertexBuffer();

		if (gfx.solverOptions.precision == 32)
		{
			this->pathlineSolver_float.Initialize(this->gfx.solverOptions);
			this->pathlineSolver_float.solve();
			this->pathlineSolver_float.FinalizeCUDA();
		}
		// under construction
		else if (gfx.solverOptions.precision == 64)
		{
		}
		this->gfx.solverOptions.beginPath = false;
		this->gfx.showLines = true;
	}

	

	if (this->gfx.solverOptions.interOperation)
	{


	}
}

void Engine::RenderFrame()
{
	gfx.RenderFrame();
}





