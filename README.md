# FLow Visualization Tool
A visualization toolset for interactive analysis of volumetical flow fields

* Installation

  + Clone the code from the repository
  + Choose the platform toolset (tested on v141 and v142)
  + Choose the Windows SDK (tested on 10.0.18362.0)
  + Set the Cuda Tookit Custom DIR (\NVIDIA GPU Computing Toolkit\CUDA\v10.2)
  + Set CUDA Include and Lib folder (e.g.NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64)
  + Set compute and sm (Tested on 6.1)
  + Add "DirectX Tool Kit" for DirectX11 to /Libs/
  + Add "DirectXTex" teture processing library to /Libs/


* Prerequisites

  + Visual Studio (tested on VS2017 and VS2019)
  + Nvidia Graphic Card with sufficient memory (dependes on the size of dataset)
  + CUDA Computing Toolkit
  + Direct3D11

* Test Dataset
  + In the Dataset UI select "Test Field"
  + It includes 20 snapshots Time-resolved incompressible turbulent channel flow at low Reynolds number provide by Davide Gatti


# External Links:
DirectX Tool Kit for DirectX 11:
https://github.com/microsoft/DirectXTK

DirectXTex texture processing library:
https://github.com/microsoft/DirectXTex


