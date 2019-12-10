#pragma once

#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/highgui/highgui.hpp> // cv::VideoCapture
#include <openpose/headers.hpp>

#include "SpoutReceiver.h"

class SpoutInput : public op::WorkerProducer<std::shared_ptr<std::vector<op::Datum>>>
{
public:
	SpoutInput(const std::string& name);
	virtual ~SpoutInput();
	void initializationOnThread();
	std::shared_ptr<std::vector<op::Datum>> workProducer();
	void printSourceNames();

private:
	// Properties
	SpoutReceiver receiver;
	char sourceName[256];
	bool isInitialized;
	uint width;
	uint height;
	cv::Mat spoutMat;

	// Methods
	bool initializeReceiver();
	void resetBuffer();
};
