
// Headers
#include <openpose/headers.hpp>
#include "TrackerPostProcessing.hpp"
#include "OscOutput.hpp"
#include "SpoutInput.hpp"

struct Config
{
	struct OSC
	{
		cv::String address;
		int port;
	} osc;
	struct Spout
	{
		cv::String name;
	} spout;
	struct Tracker
	{
		ushort persistence;
		float maxDistance;
	} tracker;
	struct OpenPose
	{
		bool renderBone;
		ushort maxTargets;
		int netWidth;
		int netHeight;
	} openpose;
};

Config loadConfig(const string& path)
{
	Config config;
	cv::FileStorage fs(path, cv::FileStorage::READ);

	if (!fs.isOpened())
	{
		op::error("Could not read config file: " + path);
		return config;
	}

	const auto osc = fs["osc"];
	cv::read(osc["address"], config.osc.address, "localhost");
	cv::read(osc["port"], config.osc.port, 7000);

	const auto spout = fs["spout"];
	cv::read(spout["name"], config.spout.name, "");

	const auto tracker = fs["tracker"];
	cv::read(tracker["persistence"], config.tracker.persistence, 10);
	cv::read(tracker["maxDistance"], config.tracker.maxDistance, 0.5);

	const auto openpose = fs["openpose"];
	cv::read(openpose["renderBone"], config.openpose.renderBone, false);
	cv::read(openpose["maxTargets"], config.openpose.maxTargets, 20);
	cv::read(openpose["netWidth"], config.openpose.netWidth, -1);
	cv::read(openpose["netHeight"], config.openpose.netHeight, 368);

	return config;
}

int spout_osc()
{
	op::ConfigureLog::setPriorityThreshold(op::Priority::High);
	op::Profiler::setDefaultX(1000);
	// // For debugging
	// // Print all logging messages
	// op::ConfigureLog::setPriorityThreshold(op::Priority::None);
	// // Print out speed values faster
	// op::Profiler::setDefaultX(100);

	const Config config = loadConfig("config.yaml");

	// 
	op::Wrapper wrapper(op::ThreadManagerMode::AsynchronousIn);

	// Input
	auto workerInput = std::make_shared<SpoutInput>(
		config.spout.name);
	wrapper.setWorker(op::WorkerType::Input, workerInput, true);
	workerInput->printSourceNames(); // Print avairalbe spout names

	// Post Processing
	auto wokerPostProcessing = std::make_shared<TrackerPostProcessing>(
		config.openpose.renderBone,
		config.tracker.persistence,
		config.tracker.maxDistance);
	wrapper.setWorker(op::WorkerType::PostProcessing, wokerPostProcessing, false);

	// Output
	auto workerOutput = std::make_shared<OscOutput>(
		config.osc.address,
		config.osc.port,
		config.openpose.renderBone);
	wrapper.setWorker(op::WorkerType::Output, workerOutput, true);

	// Pose
	const op::WrapperStructPose pose
	{
		true,
		op::Point<int>(config.openpose.netWidth, config.openpose.netHeight), // Net input size (Horizontal)
		//op::Point<int>(368, -1), // Net input size (Vertical)
		op::Point<int>(-1, -1), // Output size
		op::ScaleMode::ZeroToOne, // Keypoint scale
		-1, 0, // num_gpu, num_gpu_start (Use all GPU)
		1, // scales to average
		0.25f, // scale_gap
		config.openpose.renderBone ? op::RenderMode::Cpu : op::RenderMode::None, // render_pose
		op::PoseModel::BODY_25, // Model file
		true, // result blending
		0.6f, // Blending factor for the body part rendering
		0.7f, // Blending factor (range 0-1) between heatmap and original frame
		0, // part_to_show
		"models", // Models folder
		{}, // Heatmap types (no heatmap)
		op::ScaleMode::UnsignedChar, // Heatmap scale 
		false, // part_candidates
		0.05f, // render_threshold
		config.openpose.maxTargets, // number_people_max
		false, // maximize_positives
		-1, // fps_max
		false // enableGoogleLogging 
	};
	wrapper.configure(pose);

	//wrapper.disableMultiThreading();

	wrapper.exec();
	return 0;
}

int main(int argc, char *argv[])
{
	op::log(argv[0]);
	try
	{
		//test();
		spout_osc();
		return 0;
	}
	catch (const std::exception& e)
	{
		op::log(e.what());
		return -1;
	}
}
