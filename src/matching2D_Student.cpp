#include <numeric>
#include "matching2D.hpp"

using namespace std;

cv::Ptr<cv::Feature2D> CreateFast()
{
	auto thresshold = 30;
	auto nonmaxSupression = true;
	auto type = cv::FastFeatureDetector::TYPE_9_16;

	return cv::FastFeatureDetector::create(thresshold, nonmaxSupression, type);
}

cv::Ptr<cv::Feature2D> CreateBrisk()
{
	// Parameters from Learning OpenCV 3.0
	auto thresshold = 60;
	auto octaves = 3;
	auto patternScale = 1.0f;

	return cv::BRISK::create(thresshold, octaves, patternScale);
}

cv::Ptr<cv::Feature2D> CreateOrb()
{
	// Parameters from Learning OpenCV 3.0
	auto nFeatures = 500;
	auto scaleFactor = 1.2f;
	auto nLevels = 8;
	auto edgeThresshold = 31;
	auto firstEdge = 0; // Always 20
	auto kta_k = 2;
	auto scoreType = cv::ORB::FAST_SCORE;
	auto pathSize = 31;
	auto fastThresshold = 20;

	return cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThresshold, firstEdge, kta_k, scoreType, pathSize, fastThresshold);
}

cv::Ptr<cv::Feature2D> CreateAkaze()
{
	return cv::AKAZE::create();
}

cv::Ptr<cv::Feature2D> CreateSift()
{
	auto nFeatures = 0;
	auto nOctaveLayers = 3;
	auto contrastThreshold = 0.04;
	auto edgeThreshold = 10;
	auto sigma = 1.6;
	return cv::xfeatures2d::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

cv::Ptr<cv::Feature2D> CreateBrief()
{
	// Parameters from Learning OpenCV 3.0
	auto bytes = 32;
	auto useOrientation = false;

	return cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, useOrientation);
}

cv::Ptr<cv::Feature2D> CreateFreak()
{
	// Parameters from Learning OpenCV 3.0
	auto orientationNormalized = true;
	auto scaleNormalized = true;
	auto patternScale = 22.0f;
	auto nOctaves = 4;
	std::vector<int>  userSelectedPairs;

	return cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
	std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
	// configure matcher
	bool crossCheck = false;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	if (matcherType.compare("MAT_BF") == 0)
	{
		int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
		matcher = cv::BFMatcher::create(normType, crossCheck);
	}
	else if (matcherType.compare("MAT_FLANN") == 0)
	{
		// OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
		if (descSource.type() != CV_32F)
			descSource.convertTo(descSource, CV_32F);

		if (descRef.type() != CV_32F)
			descRef.convertTo(descRef, CV_32F);

		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	}

	// perform matching task
	if (selectorType.compare("SEL_NN") == 0)
	{ // nearest neighbor (best match)

		matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
	}
	else if (selectorType.compare("SEL_KNN") == 0)
	{ // k nearest neighbors (k=2)
		vector<vector<cv::DMatch>> knn_matches;
		matcher->knnMatch(descSource, descRef, knn_matches, 2);

		auto tresshold = 0.8;
		for (const auto& match : knn_matches) {
			if (match[0].distance < tresshold * match[1].distance)
				matches.push_back(match[0]);
		}
	}
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, string descriptorType)
{
	// select appropriate descriptor
	cv::Ptr<cv::DescriptorExtractor> extractor;
	if (descriptorType.compare("BRIEF") == 0)
	{
		extractor = CreateBrief();
	}
	else if (descriptorType.compare("ORB") == 0)
	{
		extractor = CreateOrb();
	}
	else if (descriptorType.compare("FREAK") == 0)
	{
		extractor = CreateFreak();
	}
	else if (descriptorType.compare("AKAZE") == 0)
	{
		extractor = CreateAkaze();
	}
	else if (descriptorType.compare("SIFT") == 0)
	{
		extractor = CreateSift();
	}
	else
	{
		throw exception();
	}

	// perform feature description
	double t = (double)cv::getTickCount();
	extractor->compute(img, keypoints, descriptors);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

	return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis)
{
	// compute detector parameters based on image size
	int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
	double maxOverlap = 0.0; // max. permissible overlap between two features in %
	double minDistance = (1.0 - maxOverlap) * blockSize;
	int maxCorners = static_cast<int>(img.rows * img.cols / max(1.0, minDistance)); // max. num. of keypoints

	double qualityLevel = 0.01; // minimal accepted quality of image corners
	double k = 0.04;

	// Apply corner detection
	double t = (double)cv::getTickCount();
	vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

	// add corners to result vector
	for (auto it = corners.begin(); it != corners.end(); ++it)
	{

		cv::KeyPoint newKeyPoint;
		newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
		newKeyPoint.size = static_cast<float>(blockSize);
		keypoints.push_back(newKeyPoint);
	}
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Shi-Tomasi Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

	return t;
}

double detKeypointsHarris(vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis)
{
	// compute detector parameters based on image size
	int blockSize = 2;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
	double maxOverlap = 0.0; // max. permissible overlap between two features in %
	double minDistance = (1.0 - maxOverlap) * blockSize;
	int maxCorners = static_cast<int>(img.rows * img.cols / max(1.0, minDistance)); // max. num. of keypoints

	int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
	int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;

	// Apply corner detection
	double t = (double)cv::getTickCount();

	cv::Mat dst;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

	cv::Mat dst_norm; // Normalized (0, 255) matrix
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

	// add corners to result vector
	for (auto j = 0; j < dst.rows; j++) {
		for (auto i = 0; i < dst_norm.cols; i++) {
			const auto response = static_cast<int>(dst_norm.at<float>(j, i));

			if (response < minResponse)
				continue;

			cv::KeyPoint newKeyPoint;
			newKeyPoint.pt = cv::Point2f(static_cast<float>(i), static_cast<float>(j));
			newKeyPoint.size = static_cast<float>(2 * apertureSize);
			newKeyPoint.response = static_cast<float>(response);

			// Implementamos non-maximun supression (NMS) en un vecindario local alrededor del keypoint
			auto overlap = false;
			for (auto& keypoint : keypoints) {
				if (cv::KeyPoint::overlap(newKeyPoint, keypoint) > maxOverlap) {
					// Solapan. Comprobamos si el nuevo keyPoint es mejor que el que ya teníamos contabilizado, en cuyo caso,
					// sustituimos el existente
					overlap = true;

					if (keypoint.response < newKeyPoint.response)
						keypoint = newKeyPoint;
				}
			}

			if (!overlap)
				keypoints.push_back(newKeyPoint);
		}
	}

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

	cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "HARRIS Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

	return t;
}

double detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis)
{
	cv::Ptr<cv::FeatureDetector> detector;

	if (detectorType.compare("FAST") == 0)
	{
		detector = CreateFast();
	}
	else if (detectorType.compare("BRISK") == 0)
	{
		detector = CreateBrisk();
	}
	else if (detectorType.compare("ORB") == 0)
	{
		detector = CreateOrb();
	}
	else if (detectorType.compare("AKAZE") == 0)
	{
		detector = CreateAkaze();
	}
	else if (detectorType.compare("SIFT") == 0)
	{
		detector = CreateSift();
	}
	else {
		throw exception();
	}

	double t = (double)cv::getTickCount();

	detector->detect(img, keypoints);

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType + " Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}

	return t;
}