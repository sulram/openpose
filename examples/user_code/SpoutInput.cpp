
#pragma comment (lib, "opengl32.lib")

#include <mutex> 
#include "SpoutInput.hpp"
#include <GLFW/glfw3.h>

using DatumArray = std::vector<std::shared_ptr<op::Datum>>;
using DatumArrayRef = std::shared_ptr<DatumArray>;

static GLFWwindow* window;

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

cv::Mat makeGreenMat(const int width = 1280, const int height = 720)
{
	return cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 255, 0));
}


static void logMat(const cv::Mat& mat, const std::string prepend)
{
	std::stringstream s;
	s << prepend << ": ";
	s << " size:[" << mat.cols << ", " << mat.rows << ']';
	//s << " = " << mat.size().width << ", " << mat.size().height;
	//s << " flags: " << mat.flags;
	s << " total: " << mat.total();
	s << " ch: " << mat.channels();
	s << " dep: " << mat.depth();
	s << " type: " << mat.type();
	op::opLog(s.str());
}

SpoutInput::SpoutInput(const std::string& name) :
	isInitialized{ false },
	width{ 0 },
	height{ 0 }
{
	if (!glfwInit())
	{
		op::error("Fail to initialize glfw", __LINE__, __FUNCTION__, __FILE__);
	}
	glfwSetErrorCallback(error_callback);

	if (name.size() == 0)
	{
		sourceName[0] = 0;
	}
	else
	{
		strcpy_s(sourceName, name.size() + 1, name.c_str());
	}
}

SpoutInput::~SpoutInput()
{
	if (isInitialized)
	{
		receiver.ReleaseReceiver();
	}

	spoutMat.release();
	glfwDestroyWindow(window);
	glfwTerminate();
}

void SpoutInput::initializationOnThread() {
	//receiver.SetBufferMode(true);

	// Setup OpenGL
	window = glfwCreateWindow(256, 256, "Spout Receive Window", NULL, NULL);
	if (!window)
	{
		op::error("Fail to make window", __LINE__, __FUNCTION__, __FILE__);
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}


DatumArrayRef SpoutInput::workProducer()
{

	// Create new datum
	auto datumsPtr = std::make_shared<DatumArray>();
	datumsPtr->emplace_back();
	auto& datum = datumsPtr->at(0);

	// Try Initialize
	if (!isInitialized)
	{
		isInitialized = initializeReceiver();
		datum->cvInputData = OP_CV2OPMAT(makeGreenMat());
		resetBuffer();
		return datumsPtr;
	}

	// Receive from Spout
	uint w, h;
	if (receiver.ReceiveImage(sourceName, w, h, spoutMat.ptr(), GL_BGR_EXT, false, 0)) {
		if (width != w || height != h)
		{
			width = w;
			height = h;
			resetBuffer();
		}
		else
		{
			// Need to convert if PC does not support GL_BGR_EXT
			//cv::cvtColor(spoutMat, spoutMat, CV_RGB2BGR);

			spoutMat.copyTo(OP_OP2CVMAT(datum->cvInputData));

			// This line needs when debug mode, Why???
			//cv::imshow("Spout Input", spoutMat);
		}
	}


	// Draw OpenGL
	{
		glViewport(0, 0, width, height);
		//glClear(GL_COLOR_BUFFER_BIT);
		//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	// If empty frame -> return nullptr
	if (datum->cvInputData.empty())
	{
		datum->cvInputData = OP_CV2OPMAT(makeGreenMat());
		//this->stop();
		//datumsPtr = nullptr;
	}
	return datumsPtr;
}

void SpoutInput::printSourceNames()
{
	int count = receiver.GetSenderCount();
	char name[256];
	for (int i = 0; i < count; i++)
	{
		receiver.GetSenderName(i, name);
		op::opLog("sender " + std::to_string(i) + ": " + std::string(name));
	}
}

bool SpoutInput::initializeReceiver()
{
	uint w, h;
	const bool autoSelect = sourceName[0] == 0;
	if (receiver.CreateReceiver(sourceName, w, h, autoSelect))
	{
		width = w;
		height = h;
		//glfwSetWindowSize(window, w, h);
		return true;
	}
	return false;
}

void SpoutInput::resetBuffer()
{
	spoutMat.create(height, width, CV_8UC3);
}
