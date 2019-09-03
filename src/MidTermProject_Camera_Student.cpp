/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char* argv[])
{
	/* INIT VARIABLES AND DATA STRUCTURES */

	// data location
	string dataPath = "../../../";

	// camera
	string imgBasePath = dataPath + "images/";
	string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
	string imgFileType = ".png";
	int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
	int imgEndIndex = 9;   // last file index to load
	int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

	// misc
	int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
	vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	bool bVis = false;            // visualize results

	/* MAIN LOOP OVER ALL IMAGES */

#ifdef PRINT_NUMBER_OF_KEYPOINTS
	NumberOfKeypoints numberOfKeypoints;
#endif
#ifdef PRINT_NUMBER_OF_MATCHED_KEYPOINTS
	MatchedPoints matchedPoints;
#endif
#ifdef PRINT_TIME_DETECTION_EXTRACTION
	Timers timers;
#endif

#ifdef ITERATE_ALL_DETECTORS
	for (auto detectorType : detectorTypes) {
#ifdef PRINT_NUMBER_OF_KEYPOINTS
		numberOfKeypoints[detectorType].resize(imgEndIndex - imgStartIndex + 1);
#endif

#endif
#ifdef ITERATE_ALL_DESCRIPTORS
		for (auto descriptorType : descriptorTypes) {
#ifdef PRINT_NUMBER_OF_MATCHED_KEYPOINTS
			matchedPoints[detectorType][descriptorType].resize(imgEndIndex - imgStartIndex + 1);
#endif
#ifdef PRINT_TIME_DETECTION_EXTRACTION
			timers[detectorType][descriptorType].resize(imgEndIndex - imgStartIndex + 1);
#endif
			dataBuffer.clear();

			// As you can see here, https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#akaze,
			// AKAZE descriptors can only be used with KAZE or AKAZE keypoints
			if (detectorType.compare("AKAZE") != 0 && descriptorType.compare("AKAZE") == 0)
				continue;

			// As you can see here, https://answers.opencv.org/question/5542/sift-feature-descriptor-doesnt-work-with-orb-keypoinys/?answer=13268#post-id-13268
			// SIFT tunes its OCTAVE automatically while ORB use a fixed number of octaves. I didn't try to to any of the two workaround be cause I don't want to
			// break the style of the exercise so I decided to pass this test.
			if (detectorType.compare("SIFT") == 0 && descriptorType.compare("ORB") == 0)
				continue;
#endif

			for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
			{
				/* LOAD IMAGE INTO BUFFER */

				// assemble filenames for current index
				ostringstream imgNumber;
				imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
				string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

				// load image from file and convert to grayscale
				cv::Mat img, imgGray;
				img = cv::imread(imgFullFilename);
				cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

				//// STUDENT ASSIGNMENT
				//// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
				if (dataBuffer.size() == dataBufferSize)
					dataBuffer.erase(dataBuffer.begin());

				// push image into data frame buffer
				DataFrame frame;
				frame.cameraImg = imgGray;
				dataBuffer.push_back(frame);

				//// EOF STUDENT ASSIGNMENT
				cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

				double timeKeypoints;

				/* DETECT IMAGE KEYPOINTS */

				// extract 2D keypoints from current image
				vector<cv::KeyPoint> keypoints; // create empty feature list for current image
#ifndef ITERATE_ALL_DETECTORS
				string detectorType = "SIFT";
#endif

				//// STUDENT ASSIGNMENT
				//// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
				//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

				if (detectorType.compare("SHITOMASI") == 0)
				{
					timeKeypoints = detKeypointsShiTomasi(keypoints, imgGray, false);
				}
				else if (detectorType.compare("HARRIS") == 0)
				{
					timeKeypoints = detKeypointsHarris(keypoints, imgGray, false);
				}
				else
				{
					timeKeypoints = detKeypointsModern(keypoints, imgGray, detectorType, false);
				}
				//// EOF STUDENT ASSIGNMENT


				//// STUDENT ASSIGNMENT
				//// TASK MP.3 -> only keep keypoints on the preceding vehicle

				// only keep keypoints on the preceding vehicle
				bool bFocusOnVehicle = true;
				cv::Rect vehicleRect(535, 180, 180, 150);
				if (bFocusOnVehicle)
				{
					vector<cv::KeyPoint> keypointsInVehicle;
					for (const auto& keyPoint : keypoints)
						if (vehicleRect.contains(keyPoint.pt))
							keypointsInVehicle.push_back(keyPoint);

					keypoints = keypointsInVehicle;
				}

#ifdef PRINT_NUMBER_OF_KEYPOINTS
				numberOfKeypoints[detectorType][imgIndex] = keypoints.size();
#endif

				//// EOF STUDENT ASSIGNMENT

				// optional : limit number of keypoints (helpful for debugging and learning)
				bool bLimitKpts = false;
				if (bLimitKpts)
				{
					int maxKeypoints = 50;

					if (detectorType.compare("SHITOMASI") == 0)
					{ // there is no response info, so keep the first 50 as they are sorted in descending quality order
						keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
					}
					cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
					cout << " NOTE: Keypoints have been limited!" << endl;
				}

				// push keypoints and descriptor for current frame to end of data buffer
				(dataBuffer.end() - 1)->keypoints = keypoints;
				cout << "#2 : DETECT KEYPOINTS done" << endl;

				/* EXTRACT KEYPOINT DESCRIPTORS */

				//// STUDENT ASSIGNMENT
				//// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
				//// -> BRIEF, ORB, FREAK, AKAZE, SIFT

				cv::Mat descriptors;
#ifndef ITERATE_ALL_DESCRIPTORS
				string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
#endif
				auto timeDescriptorExtractor = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

#ifdef PRINT_TIME_DETECTION_EXTRACTION
				timers[detectorType][descriptorType][imgIndex] = make_tuple(1000 * timeKeypoints, 100 * timeDescriptorExtractor);
#endif

				//// EOF STUDENT ASSIGNMENT

				// push descriptors for current frame to end of data buffer
				(dataBuffer.end() - 1)->descriptors = descriptors;

				cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

				if (dataBuffer.size() > 1) // wait until at least two images have been processed
				{

					/* MATCH KEYPOINT DESCRIPTORS */

					vector<cv::DMatch> matches;
					string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
					string _descriptorType = 0 == descriptorType.compare("SIFT") ? "DES_HOG" : "DES_BINARY";
					string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

					//// STUDENT ASSIGNMENT
					//// TASK MP.5 -> add FLANN matching in file matching2D.cpp
					//// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

					matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
						(dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
						matches, _descriptorType, matcherType, selectorType);

					//// EOF STUDENT ASSIGNMENT

					// store matches in current data frame
					(dataBuffer.end() - 1)->kptMatches = matches;

#ifdef PRINT_NUMBER_OF_MATCHED_KEYPOINTS
					matchedPoints[detectorType][descriptorType][imgIndex - 1] = matches.size();
#endif
					cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

					// visualize matches between current and previous image
#ifdef SHOW_IMAGES
					bVis = true;
					if (bVis)
					{
						cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
						cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
							(dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
							matches, matchImg,
							cv::Scalar::all(-1), cv::Scalar::all(-1),
							vector<char>(),
#ifdef PERFORMANCE_EVALUATION_1
							cv::DrawMatchesFlags::DEFAULT
#else
							cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
						);

						string windowName = "Matching keypoints between two camera images";
						cv::namedWindow(windowName, 7);
						cv::imshow(windowName, matchImg);
						cout << "Press key to continue to next image" << endl;
						cv::waitKey(0); // wait for key to be pressed
					}
					bVis = false;
#endif
				}
			} // eof loop over all images
#ifdef ITERATE_ALL_DESCRIPTORS
		}
#endif
#ifdef ITERATE_ALL_DETECTORS
	}

#ifdef PRINT_NUMBER_OF_KEYPOINTS
	for (auto detectorType : detectorTypes) {
		cout << "__" << detectorType << "__" << endl;

		cout << "|Image|Keypoints|" << endl;
		cout << "|---|---|" << endl;

		for (auto i = 0; i < numberOfKeypoints[detectorType].size(); i++)
			cout << "|" << 1 + i << "|" << numberOfKeypoints[detectorType][i] << "|" << endl;
	}
#endif

#ifdef PRINT_NUMBER_OF_MATCHED_KEYPOINTS
	cout << "<table>";
	cout << "<tr>";
	cout << "<th></th><th>Image</th>";

	for (auto descriptorType : descriptorTypes)
		cout << "<th>" << descriptorType << "</th>";
	cout << "</tr>";

	map<string, map<string, size_t>> totalMatches;

	for (auto detectorType : detectorTypes) {
		cout << "<tr><td rowspan=\"" << imgEndIndex - imgStartIndex + 2 << "\">" << detectorType << "</td></tr>";

		for (size_t imgIndex = 0; imgIndex < imgEndIndex - imgStartIndex; imgIndex++)
		{
			cout << "<tr>";
			cout << "<td>" << 1 + imgIndex << "</td>";

			for (auto descriptorType : descriptorTypes) {
				cout << "<td>";

				auto value = matchedPoints[detectorType][descriptorType][imgIndex];
				totalMatches[detectorType][descriptorType] += value;

				if (value)
					cout << value;

				cout << "</td>";
			}
			cout << "</tr>";
		}

		cout << "<tr>";
		cout << "<td><strong>Total</strong></td>";

		for (auto descriptorType : descriptorTypes) {
			cout << "<td><strong>";

			size_t total = totalMatches[detectorType][descriptorType];

			if (total)
				cout << total;

			cout << "</strong></td>";
		}
		cout << "</tr>";
	}
	cout << "</table>" << endl << endl;
#endif

#ifdef PRINT_TIME_DETECTION_EXTRACTION
	cout << fixed;
	cout << "<table>";
	cout << "<tr>";
	cout << "<th></th><th>Image</th>";

	for (auto descriptorType : descriptorTypes)
		cout << "<th>" << descriptorType << "</th>";
	cout << "</tr>";

	map<string, map<string, double>> totalTimes;

	for (auto detectorType : detectorTypes) {
		cout << "<tr><td rowspan=\"" << imgEndIndex - imgStartIndex + 3 << "\">" << detectorType << "</td></tr>";

		for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
		{
			cout << "<tr>";
			cout << "<td>" << 1 + imgIndex << "</td>";

			for (auto descriptorType : descriptorTypes) {
				auto timeKeypoint = get<0>(timers[detectorType][descriptorType][imgIndex]);
				auto timeDescriptor = get<1>(timers[detectorType][descriptorType][imgIndex]);
				totalTimes[detectorType][descriptorType] += timeKeypoint + timeDescriptor;

				cout << "<td>";

				if (0 != timeKeypoint || 0 != timeDescriptor)
					cout << setprecision(1) << get<0>(timers[detectorType][descriptorType][imgIndex]) << "+" << setprecision(1) << get<1>(timers[detectorType][descriptorType][imgIndex]);

				cout << "</td>";
			}
			cout << "</tr>";
		}

		cout << "<tr>";
		cout << "<td><strong>Total</strong></td>";

		for (auto descriptorType : descriptorTypes) {
			cout << "<td><strong>";

			if (0 != totalTimes[detectorType][descriptorType])
				cout << setprecision(1) << totalTimes[detectorType][descriptorType];

			cout << "</td></strong>";
		}
		cout << "</tr>";

	}
	cout << "</table>" << endl << endl;
#endif

#if defined(PRINT_NUMBER_OF_MATCHED_KEYPOINTS) && defined(PRINT_TIME_DETECTION_EXTRACTION)
	// Print performance (number of matches per milisecond)
	cout << "<table>";
	cout << "<tr>";
	cout << "<th></th>";

	for (auto descriptorType : descriptorTypes)
		cout << "<th>" << descriptorType << "</th>";
	cout << "</tr>";

	for (auto detectorType : detectorTypes) {
		cout << "<tr><td>" << detectorType << "</td>";

		for (auto descriptorType : descriptorTypes) {
			cout << "<td>";

			auto matches = totalMatches[detectorType][descriptorType];
			auto time = totalTimes[detectorType][descriptorType];

			if (time != 0)
				cout << setprecision(1) << matches / time;

			cout << "</td>";
		}

		cout << "</tr>";
	}
	cout << "</table>" << endl << endl;
#endif
#endif

	return 0;
}
