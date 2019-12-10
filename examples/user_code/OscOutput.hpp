#pragma once

#include <iostream>
#include <asio.hpp>
#include <oscpp/client.hpp>
#include <openpose/headers.hpp>


class OscOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<op::Datum>>>
{

public:

	OscOutput(const std::string& host, const int& port, const bool showImage) :
		service{},
		socket{ service, asio::ip::udp::endpoint(asio::ip::udp::v4(), 0) },
		showImage{ showImage }
	{
		using asio::ip::udp;

		udp::resolver resolver{ service };
		udp::resolver::query query{ udp::v4(), host, std::to_string(port) };
		udp::resolver::iterator iter = resolver.resolve(query);
		endpoint = *iter;
	}

	virtual ~OscOutput() {
		socket.close();
	}

	void initializationOnThread() {}

	void workConsumer(const std::shared_ptr<std::vector<op::Datum>>& datumsPtr)
	{
		try {
			if (datumsPtr == nullptr || datumsPtr->empty()) {
				// No human in the image
				return;
			}

			sendOSC(datumsPtr->at(0));

			if (showImage)
			{
				// Display rendered output image
				cv::Mat mat = datumsPtr->at(0).cvOutputData;
				if (!mat.empty())
				{
					cv::imshow("Osc Output", mat);
				}

				// Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
				const char key = (char)cv::waitKey(1);
				if (key == 27) // Escape to quit
				{
					this->stop();
				}
			}
		}
		catch (const std::exception& e) {
			//this->stop();
			op::log(e.what());
			op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

private:
	asio::io_service service;
	asio::ip::udp::socket socket;
	asio::ip::udp::endpoint endpoint;
	char buffer[8192];
	bool showImage = false;

	void sendOSC(const op::Datum& datum)
	{
		const auto& keypoints = datum.poseKeypoints;

		// Accesing each element of the keypoints
		const int persons = keypoints.getSize(0);
		const int parts = keypoints.getSize(1);
		const int dimensions = keypoints.getSize(2) - 1; // ignore score data
		if (dimensions != 2)
		{
			// No person
			return;
		}
		const auto& ids = datum.poseIds;

		constexpr uint PACKET_LIMIT = 8192 - 276; // 276 is one skelten packets

		// Start OSC bundle
		OSCPP::Client::Packet packet(buffer, sizeof(buffer));
		packet.openBundle(1234ULL);

		for (int person = 0; person < persons; person++)
		{
			packet.openMessage("/skeleton", 1 + parts * dimensions);
			packet.int32(ids[{0, person}]);

			for (int part = 0; part < parts; part++)
			{
				// Should Send all info
				//for (int dim = 0; dim < dimensions; dim++) {
				//	packet.float32(poseKeypoints[{person, part, dim}]);
				//}
				// invert Y
				packet.float32(keypoints[{person, part, 0}]);
				packet.float32(1.f - keypoints[{person, part, 1}]); // invert Y
			}
			packet.closeMessage();

			if (packet.size() >= PACKET_LIMIT) {
				break;
			}
		}

		packet.closeBundle();
		socket.send_to(asio::buffer(buffer, packet.size()), endpoint);
	}
};
