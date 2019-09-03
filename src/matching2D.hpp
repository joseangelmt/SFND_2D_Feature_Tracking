#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

#define PERFORMANCE_EVALUATION


#ifdef PERFORMANCE_EVALUATION
#define PRINT_NUMBER_OF_KEYPOINTS
#define PRINT_NUMBER_OF_MATCHED_KEYPOINTS
#define PRINT_TIME_DETECTION_EXTRACTION
#define ITERATE_ALL_DETECTORS
#define ITERATE_ALL_DESCRIPTORS
#endif

#ifdef PRINT_NUMBER_OF_KEYPOINTS
using NumberOfKeypoints = std::map<std::string, std::vector<std::size_t>>;
#endif

#ifdef PRINT_NUMBER_OF_MATCHED_KEYPOINTS
using MatchedPoints = std::map<std::string, std::map<std::string, std::vector<std::size_t>>>;
#endif

#ifdef PRINT_TIME_DETECTION_EXTRACTION
using Timers = std::map<std::string, std::map<std::string, std::vector<std::tuple<double, double>>>>;
#endif

#ifdef ITERATE_ALL_DETECTORS
static std::vector<std::string> detectorTypes{
	"SHITOMASI",
	"HARRIS",
	"FAST",
	"BRISK",
	"ORB",
	"AKAZE",
	"SIFT"
};
#endif

#ifdef ITERATE_ALL_DESCRIPTORS
static std::vector<std::string> descriptorTypes{
	"BRIEF",
	"ORB",
	"FREAK",
	"AKAZE",
	"SIFT"
};
#endif

#ifndef PERFORMANCE_EVALUATION
#define SHOW_IMAGES
#endif

double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
double detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);
double descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
