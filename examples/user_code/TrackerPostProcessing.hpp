#pragma once

#include <openpose/headers.hpp>
#include "Tracker.h"

class TrackerPostProcessing : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
	struct Target
	{
		cv::Point2f center;
		unsigned long id;
		unsigned long updated; // milli seconds
	};

public:
    TrackerPostProcessing(bool drawInfo, ushort persistence, float maxDistance);
	virtual ~TrackerPostProcessing();

    void initializationOnThread();

    void work(std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr);

private:
	bool drawInfo;
	ofxCv::Rect2fTrackerFollower<ofxCv::Rect2fFollower> tracker;

};
