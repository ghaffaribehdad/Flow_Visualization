#pragma once
#include <string>

struct FieldOptions
{
	std::string fileName;
	std::string filePath;
	std::string subpath;
	int gridSize[3] = { 0,0,0 };
	float gridDiameter[3] = { 0,0,0 };
	bool isCompressed = false;
	bool isEnsemble = false;
	int firstMemberIdx = 0;
	int lastMemberIdx = 0;
	int currentIdx = 1;
	int firstIdx = 0;
	int lastIdx = 0;
	float dt = 0;
	size_t fileSizeMaxByte = 0;


	void setField(
		std::string _fileName,
		std::string _filePath,
		int _gridSizeX,
		int _gridSizeY,
		int _gridSizeZ,
		float _gridDiameterX,
		float _gridDiameterY,
		float _gridDiameterZ,
		int _firstIdx,
		int _lastIdx,
		float _dt,
		bool _isCompressed = false,
		size_t _fileSizeMaxByte = 0
	)
	{
		fileName = _fileName;
		filePath = _filePath;

		gridSize[0] = _gridSizeX;
		gridSize[1] = _gridSizeY;
		gridSize[2] = _gridSizeZ;

		gridDiameter[0] = _gridDiameterX;
		gridDiameter[1] = _gridDiameterY;
		gridDiameter[2] = _gridDiameterZ;

		firstIdx	= _firstIdx;
		lastIdx		= _lastIdx;
		dt = _dt;
		isCompressed = _isCompressed;
		fileSizeMaxByte = _fileSizeMaxByte;

	}

	void setFieldEnsemble(
		std::string _fileName,
		std::string _filePath,
		std::string _subPath,
		int _gridSizeX,
		int _gridSizeY,
		int _gridSizeZ,
		float _gridDiameterX,
		float _gridDiameterY,
		float _gridDiameterZ,
		int _firstIdx,
		int _lastIdx,
		int _firstMemberIdX,
		int _lastMemberIdX,
		float _dt,
		bool _isCompressed = false,
		size_t _fileSizeMaxByte = 0
		)
	{
		setField(_fileName, _filePath, _gridSizeX, _gridSizeY, _gridSizeZ, _gridDiameterX, _gridDiameterY, _gridDiameterZ, _firstIdx, _lastIdx, _dt,_isCompressed,_fileSizeMaxByte);
		subpath = _subPath;
		firstMemberIdx = _firstMemberIdX;
		lastMemberIdx = _lastMemberIdX;
		isEnsemble = true;
	}

};