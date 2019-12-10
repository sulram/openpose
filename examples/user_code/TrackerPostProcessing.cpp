#include <algorithm>		
#include <vector>
#include <limits>

#include "TrackerPostProcessing.hpp"

using namespace std::chrono;

// Implements simple centroid tracking
using DatumArray = std::vector<std::shared_ptr<op::Datum>>;
using DatumArrayRef = std::shared_ptr<DatumArray>;

namespace ofxCv
{
	float trackingDistance(const cv::Rect& a, const cv::Rect& b)
	{
		float dx = (a.x + a.width / 2.) - (b.x + b.width / 2.);
		float dy = (a.y + a.height / 2.) - (b.y + b.height / 2.);
		float dw = a.width - b.width;
		float dh = a.height - b.height;
		float pd = sqrtf(dx * dx + dy * dy);
		float sd = sqrtf(dw * dw + dh * dh);
		return pd + sd;
	}

	float trackingDistance(const cv::Rect2f &a, const cv::Rect2f& b)
	{
		float dx = (a.x + a.width / 2.) - (b.x + b.width / 2.);
		float dy = (a.y + a.height / 2.) - (b.y + b.height / 2.);
		float dw = a.width - b.width;
		float dh = a.height - b.height;
		float pd = sqrtf(dx * dx + dy * dy);
		float sd = sqrtf(dw * dw + dh * dh);
		return pd + sd;
	}

	float trackingDistance(const cv::Point2f& a, const cv::Point2f& b)
	{
		float dx = a.x - b.x;
		float dy = a.y - b.y;
		return sqrtf(dx * dx + dy * dy);
	}
}

inline cv::Rect2f boundingRect(const op::Array<float>& keypoints, int person)
{
	// since cv::boundingRect() returns intger
	assert(person < keypoints.getSize(0));

	const int parts = keypoints.getSize(1);

	float minX = std::numeric_limits<float>::max();
	float minY = std::numeric_limits<float>::max();
	float maxX = std::numeric_limits<float>::min();
	float maxY = std::numeric_limits<float>::min();

	for (int part = 0; part < parts; part++)
	{
		const float x = keypoints[{person, part, 0}];
		const float y = keypoints[{person, part, 1}];
		if (x > 0 || y > 0)
		{
			minX = std::min(minX, x);
			minY = std::min(minY, y);
			maxX = std::max(maxX, x);
			maxY = std::max(maxY, y);
		}
	}
	return cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
}


TrackerPostProcessing::TrackerPostProcessing(bool drawInfo, ushort persistence, float maxDistance) :
	drawInfo{ drawInfo }
{
	tracker.setPersistence(persistence);
	tracker.setMaximumDistance(maxDistance);
}

TrackerPostProcessing::~TrackerPostProcessing()
{
}

void TrackerPostProcessing::initializationOnThread()
{
}

void TrackerPostProcessing::work(DatumArrayRef& datumsPtr)
{

	if (datumsPtr == nullptr || datumsPtr->empty())
	{
		// No human in the image
		// no thread queue
		return;
	}



	// Process all queue
	for (auto& datum : *datumsPtr)
	{
		const auto& keypoints = datum->poseKeypoints;
		const int persons = keypoints.getSize(0);
		//const int parts = keypoints.getSize(1);
		const int dimensions = keypoints.getSize(2); // ignore score data
		if (dimensions < 3) {
			continue;
		}

		// Track
		std::vector<cv::Rect2f> bboxes;
		for (int i = 0; i < persons; i++)
		{
			bboxes.push_back(boundingRect(keypoints, i));
		}
		tracker.track(bboxes);

		// Assign IDs
		op::Array<long long> ids{ persons, -1 };
		for (int i = 0; i < persons; i++)
		{
			ids[{0, i}] = tracker.getLabelFromIndex(i);
		}
		datum->poseIds = ids;

		if (drawInfo)
		{
			cv::Mat& image = OP_OP2CVMAT(datum->cvOutputData);
			const cv::Size2f size(image.cols, image.rows);

			for (int i = 0; i < persons; i++)
			{
				// Draw debug text
				const auto& bbox = bboxes.at(i);
				const cv::Point2f p = (bbox.tl() + bbox.br()) / 2;
				cv::putText(image,
					std::to_string(ids[{0, i}]),
					cv::Point2f(p.x * size.width, p.y * size.height),
					cv::FONT_HERSHEY_PLAIN, 1.0, { 0, 255, 0 });
			}
		}

	}

}
