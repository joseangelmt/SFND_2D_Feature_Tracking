# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

# Mid-Term Project Submission

## Data Buffer
### MP.1 Data Buffer Optimization

_Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end._

I solved it with the next code that deletes the first item from the list when it's size is already `dataBufferSize`:

```c++
if (dataBuffer.size() == dataBufferSize)
    dataBuffer.erase(dataBuffer.begin());
```

## Keypoints
### MP.2 Keypoint Detection

_Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly._

I added two functions: `detKeypointsHarris` for _HARRIS_ and `detKeypointsModern` for the rest of detectors.

For the _modern_ detectors, the function `detKeypointsModern` creates the detector using polymorphism (all modern detectors inherits from _cv::Feature2D_ and actually _cv::FeatureDetector_ and _cv::DescriptorExtractor_ are an alias of _cv::Feature2D_) and then uses it like this:

```c++
void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis)
{
	cv::Ptr<cv::FeatureDetector> detector;

	if (detectorType.compare("FAST") == 0)
	{
		detector = Create detector FAST
	}
	else if (detectorType.compare("...") == 0)
	{
		detector = Create detector ...
	}
        ...
    
	double t = (double)cv::getTickCount();

	detector->detect(img, keypoints);

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

	// visualize results
	if (bVis)
	{
		...
	}
}
```

Then I created a function for each detector like this:

```c++
cv::Ptr<cv::Feature2D> CreateFast()
{
	auto thresshold = 30;
	auto nonmaxSupression = true;
	auto type = cv::FastFeatureDetector::TYPE_9_16;

	return cv::FastFeatureDetector::create(thresshold, nonmaxSupression, type);
}
```

The parameters I used for the detectors are in the next tables:

__HARRIS__

|Parameter|Value|
|---|---|
|Block size|2|
|Aperture size|3|
|Min response|100|
|k|0.04|

__FAST__

|Parameter|Value|
|---|---|
|Thresshold|30|
|Non Maximal Supression|true|
|Type|9_16|

__BRISK__

|Parameter|Value|
|---|---|
|Thresshold|30|
|Octaves|3|
|Pattern scale|1.0f|

__ORB__

|Parameter|Value|
|---|---|
|Maximum features to compute|500|
|Pyramid ratio|1.2|
|Number of pyramid levels to use|8|
|Size of no-search border|31|
|First level|0|
|Pts in each comparison|2|
|Score type|FAST|
|Size of patch for each descriptor|31|
|Threshold for FAST detector|20|

__AKAZE__

Not adjustable parameters for AKAZE

__SIFT__

|Parameter|Value|
|---|---|
|Number of features to use|0|
|Layers in each octave|3|
|Contrast Thresshold|0.04|
|Edge Thresshold|10|
|Sigma|1.6|

### MP.3 Keypoint Removal
_Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing._

I solved it using the `cv::Rect::contains` method creating a new vector of _KeyPoints_ that have only the points inside the rectangle and then assigning the new vector to _keypoints_ like this:

```c++
if (bFocusOnVehicle)
{
	vector<cv::KeyPoint> keypointsInVehicle;
	for (const auto& keyPoint : keypoints)
		if (vehicleRect.contains(keyPoint.pt))
			keypointsInVehicle.push_back(keyPoint);

	keypoints = keypointsInVehicle;
}
```

## Descriptors
### MP.4 Keypoint Descriptors
_Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly._

I did the same schema of function `detKeypointsModern` in task _MP.2 Keypoint Detection_, so I added the functions to create the detectors _BRIEF_ and _FREAK_ with the next parameters:

__BRIEF__

|Parameter|Value|
|---|---|
|Bytes|32|
|Use orientation|false|

__FREAK__

|Parameter|Value|
|---|---|
|Orientation normalized|true|
|Scale normalized|true|
|Pattern scale|22.0f|
|Number of octaves|4|

### MP.5 Descriptor Matching
_Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function._

I added the workaround to solve the floating point error for both images: source and ref and then I instantiated the matcher like this:

```c++
// OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
if (descSource.type() != CV_32F)
	descSource.convertTo(descSource, CV_32F);

if (descRef.type() != CV_32F)
	descRef.convertTo(descRef, CV_32F);

matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
```

### MP.6 Descriptor Distance Ratio
_Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints._

I used the implementation I did in _Lesson 4: 5. Descriptor Matching_ like this:

```c++
vector<vector<cv::DMatch>> knn_matches;
matcher->knnMatch(descSource, descRef, knn_matches, 2);

auto tresshold = 0.8;
for (const auto& match : knn_matches) {
	if (match[0].distance < tresshold * match[1].distance)
		matches.push_back(match[0]);
}
```

## Performance

I have modified the source code of the program so that if the `PERFORMANCE_EVALUATION` macro is defined in the file _matching2D.hpp_, the program iterates all detectors/descriptors/images and prints at the end of the execution the tables of the following three sub-sections.

### MP.7 Performance Evaluation 1
_Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented._

__Shi-Tomasi__

|Image|Keypoints|
|---|---|
|1|125|
|2|118|
|3|123|
|4|120|
|5|120|
|6|113|
|7|114|
|8|123|
|9|111|
|10|112|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|Yes|Yes|
|Lights|Yes|Yes|
|Rear-view mirrors|Yes|Yes|

__HARRIS__

|Image|Keypoints|
|---|---|
|1|17|
|2|14|
|3|19|
|4|22|
|5|26|
|6|47|
|7|18|
|8|33|
|9|27|
|10|35|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|No|N/A|
|Lights|Yes|Yes|
|Rear-view mirrors|No|N/A|

__FAST__

|Image|Keypoints|
|---|---|
|1|149|
|2|152|
|3|150|
|4|155|
|5|149|
|6|149|
|7|156|
|8|150|
|9|138|
|10|143|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|Yes|No|
|Lights|Yes|Yes|
|Rear-view mirrors|Yes|Yes|

__BRISK__

|Image|Keypoints|
|---|---|
|1|158|
|2|162|
|3|155|
|4|161|
|5|158|
|6|155|
|7|157|
|8|151|
|9|157|
|10|138|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|Yes|No|
|Lights|Yes|Yes|
|Rear-view mirrors|Yes|Yes|

__ORB__

|Image|Keypoints|
|---|---|
|1|89|
|2|91|
|3|92|
|4|94|
|5|103|
|6|103|
|7|110|
|8|114|
|9|105|
|10|102|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|No|N/A|
|Lights|Yes|Yes|
|Rear-view mirrors|No|N/A|

__AKAZE__

|Image|Keypoints|
|---|---|
|1|166|
|2|157|
|3|161|
|4|155|
|5|163|
|6|164|
|7|173|
|8|175|
|9|177|
|10|179|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|No|N/A|
|Lights|Yes|Yes|
|Rear-view mirrors|Yes|Yes|

__SIFT__

|Image|Keypoints|
|---|---|
|1|138|
|2|132|
|3|124|
|4|137|
|5|134|
|6|140|
|7|137|
|8|148|
|9|159|
|10|137|

Distribution:

|Zone of interest|Keypoints|Repeatability|
|---|---|---|
|Back windscreen|Yes|Yes|
|Plate|Yes|Yes|
|Lights|Yes|Yes|
|Rear-view mirrors|Yes|Yes|

### MP.8 Performance Evaluation 2
_Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8._

First of all, I had to add logic to enable add compatilibity with SIFT to the Brute Force Matcher like this:

```c++
if (matcherType.compare("MAT_BF") == 0)
{
	int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
	matcher = cv::BFMatcher::create(normType, crossCheck);
}
```

Executing this test found two incompatibilities.

1. AKAZE descriptors can only be used with KAZE or AKAZE keypoints [as you can see here](https://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#akaze).
2. SIFT keypoints are not compatible with ORB descriptors [as you can see here](https://answers.opencv.org/question/5542/sift-feature-descriptor-doesnt-work-with-orb-keypoinys/?answer=13268#post-id-13268). I know there is a workaround but I didn't develop it because I don't want to break the style of the project passing additional parameters to the functions.

<table><tr><th></th><th>Image</th><th>BRIEF</th><th>ORB</th><th>FREAK</th><th>AKAZE</th><th>SIFT</th></tr><tr><td rowspan="11">SHITOMASI</td></tr><tr><td>1</td><td>115</td><td>106</td><td>86</td><td></td><td>112</td></tr><tr><td>2</td><td>111</td><td>102</td><td>90</td><td></td><td>109</td></tr><tr><td>3</td><td>104</td><td>99</td><td>86</td><td></td><td>104</td></tr><tr><td>4</td><td>101</td><td>102</td><td>88</td><td></td><td>103</td></tr><tr><td>5</td><td>102</td><td>103</td><td>86</td><td></td><td>99</td></tr><tr><td>6</td><td>102</td><td>97</td><td>80</td><td></td><td>101</td></tr><tr><td>7</td><td>100</td><td>98</td><td>81</td><td></td><td>96</td></tr><tr><td>8</td><td>109</td><td>104</td><td>86</td><td></td><td>106</td></tr><tr><td>9</td><td>100</td><td>97</td><td>85</td><td></td><td>97</td></tr><tr><td><strong>Total</strong></td><td><strong>944</strong></td><td><strong>908</strong></td><td><strong>768</strong></td><td><strong></strong></td><td><strong>927</strong></td></tr><tr><td rowspan="11">HARRIS</td></tr><tr><td>1</td><td>14</td><td>12</td><td>13</td><td></td><td>14</td></tr><tr><td>2</td><td>11</td><td>12</td><td>11</td><td></td><td>11</td></tr><tr><td>3</td><td>16</td><td>15</td><td>15</td><td></td><td>16</td></tr><tr><td>4</td><td>21</td><td>19</td><td>16</td><td></td><td>20</td></tr><tr><td>5</td><td>23</td><td>23</td><td>16</td><td></td><td>21</td></tr><tr><td>6</td><td>28</td><td>21</td><td>21</td><td></td><td>23</td></tr><tr><td>7</td><td>15</td><td>14</td><td>12</td><td></td><td>13</td></tr><tr><td>8</td><td>23</td><td>23</td><td>19</td><td></td><td>22</td></tr><tr><td>9</td><td>23</td><td>22</td><td>18</td><td></td><td>22</td></tr><tr><td><strong>Total</strong></td><td><strong>174</strong></td><td><strong>161</strong></td><td><strong>141</strong></td><td><strong></strong></td><td><strong>162</strong></td></tr><tr><td rowspan="11">FAST</td></tr><tr><td>1</td><td>119</td><td>118</td><td>98</td><td></td><td>118</td></tr><tr><td>2</td><td>130</td><td>123</td><td>99</td><td></td><td>123</td></tr><tr><td>3</td><td>118</td><td>112</td><td>91</td><td></td><td>110</td></tr><tr><td>4</td><td>126</td><td>126</td><td>98</td><td></td><td>119</td></tr><tr><td>5</td><td>108</td><td>106</td><td>85</td><td></td><td>114</td></tr><tr><td>6</td><td>123</td><td>122</td><td>99</td><td></td><td>119</td></tr><tr><td>7</td><td>131</td><td>122</td><td>102</td><td></td><td>123</td></tr><tr><td>8</td><td>125</td><td>123</td><td>101</td><td></td><td>117</td></tr><tr><td>9</td><td>119</td><td>119</td><td>105</td><td></td><td>103</td></tr><tr><td><strong>Total</strong></td><td><strong>1099</strong></td><td><strong>1071</strong></td><td><strong>878</strong></td><td><strong></strong></td><td><strong>1046</strong></td></tr><tr><td rowspan="11">BRISK</td></tr><tr><td>1</td><td>111</td><td>104</td><td>110</td><td></td><td>113</td></tr><tr><td>2</td><td>119</td><td>104</td><td>109</td><td></td><td>117</td></tr><tr><td>3</td><td>102</td><td>100</td><td>99</td><td></td><td>108</td></tr><tr><td>4</td><td>101</td><td>107</td><td>110</td><td></td><td>114</td></tr><tr><td>5</td><td>104</td><td>94</td><td>102</td><td></td><td>102</td></tr><tr><td>6</td><td>110</td><td>109</td><td>107</td><td></td><td>112</td></tr><tr><td>7</td><td>111</td><td>96</td><td>98</td><td></td><td>100</td></tr><tr><td>8</td><td>106</td><td>103</td><td>98</td><td></td><td>108</td></tr><tr><td>9</td><td>118</td><td>101</td><td>104</td><td></td><td>112</td></tr><tr><td><strong>Total</strong></td><td><strong>982</strong></td><td><strong>918</strong></td><td><strong>937</strong></td><td><strong></strong></td><td><strong>986</strong></td></tr><tr><td rowspan="11">ORB</td></tr><tr><td>1</td><td>52</td><td>63</td><td>41</td><td></td><td>67</td></tr><tr><td>2</td><td>36</td><td>69</td><td>39</td><td></td><td>73</td></tr><tr><td>3</td><td>39</td><td>59</td><td>44</td><td></td><td>67</td></tr><tr><td>4</td><td>46</td><td>71</td><td>46</td><td></td><td>72</td></tr><tr><td>5</td><td>46</td><td>82</td><td>41</td><td></td><td>74</td></tr><tr><td>6</td><td>58</td><td>84</td><td>42</td><td></td><td>83</td></tr><tr><td>7</td><td>58</td><td>76</td><td>44</td><td></td><td>86</td></tr><tr><td>8</td><td>54</td><td>85</td><td>48</td><td></td><td>84</td></tr><tr><td>9</td><td>57</td><td>79</td><td>51</td><td></td><td>87</td></tr><tr><td><strong>Total</strong></td><td><strong>446</strong></td><td><strong>668</strong></td><td><strong>396</strong></td><td><strong></strong></td><td><strong>693</strong></td></tr><tr><td rowspan="11">AKAZE</td></tr><tr><td>1</td><td>141</td><td>131</td><td>126</td><td>138</td><td>134</td></tr><tr><td>2</td><td>134</td><td>129</td><td>129</td><td>138</td><td>134</td></tr><tr><td>3</td><td>131</td><td>127</td><td>127</td><td>133</td><td>130</td></tr><tr><td>4</td><td>130</td><td>117</td><td>121</td><td>127</td><td>136</td></tr><tr><td>5</td><td>134</td><td>130</td><td>122</td><td>129</td><td>137</td></tr><tr><td>6</td><td>146</td><td>131</td><td>133</td><td>146</td><td>147</td></tr><tr><td>7</td><td>150</td><td>137</td><td>144</td><td>147</td><td>147</td></tr><tr><td>8</td><td>148</td><td>135</td><td>147</td><td>151</td><td>154</td></tr><tr><td>9</td><td>152</td><td>145</td><td>138</td><td>150</td><td>151</td></tr><tr><td><strong>Total</strong></td><td><strong>1266</strong></td><td><strong>1182</strong></td><td><strong>1187</strong></td><td><strong>1259</strong></td><td><strong>1270</strong></td></tr><tr><td rowspan="11">SIFT</td></tr><tr><td>1</td><td>86</td><td></td><td>65</td><td></td><td>82</td></tr><tr><td>2</td><td>78</td><td></td><td>72</td><td></td><td>81</td></tr><tr><td>3</td><td>76</td><td></td><td>64</td><td></td><td>85</td></tr><tr><td>4</td><td>85</td><td></td><td>66</td><td></td><td>93</td></tr><tr><td>5</td><td>69</td><td></td><td>59</td><td></td><td>90</td></tr><tr><td>6</td><td>74</td><td></td><td>59</td><td></td><td>81</td></tr><tr><td>7</td><td>76</td><td></td><td>64</td><td></td><td>82</td></tr><tr><td>8</td><td>70</td><td></td><td>65</td><td></td><td>102</td></tr><tr><td>9</td><td>88</td><td></td><td>79</td><td></td><td>104</td></tr><tr><td><strong>Total</strong></td><td><strong>702</strong></td><td><strong></strong></td><td><strong>593</strong></td><td><strong></strong></td><td><strong>800</strong></td></tr></table>

### MP.9 Performance Evaluation 3
_Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles._

For to count the times, I had to modify the signature of the functions _detKeypointsHarris_, _detKeypointsShiTomasi_, _detKeypointsModern_, _descKeypoints_ to return the time in seconds instead of `void`.

I have printed in the following table two times: The time for keypoint detection plus the time for descriptor extraction for each image. At the end I show the total time in all the images for a certain combination of keypoint detector/descriptor extraction.

<table><tr><th></th><th>Image</th><th>BRIEF</th><th>ORB</th><th>FREAK</th><th>AKAZE</th><th>SIFT</th></tr><tr><td rowspan="12">SHITOMASI</td></tr><tr><td>1</td><td>44.8+0.7</td><td>20.8+0.2</td><td>20.0+4.7</td><td></td><td>16.8+2.3</td></tr><tr><td>2</td><td>19.0+0.2</td><td>18.9+0.2</td><td>21.6+5.2</td><td></td><td>15.7+1.9</td></tr><tr><td>3</td><td>22.3+0.2</td><td>22.5+0.2</td><td>17.1+5.6</td><td></td><td>15.4+2.1</td></tr><tr><td>4</td><td>19.4+0.2</td><td>16.7+0.2</td><td>15.4+5.5</td><td></td><td>24.6+3.0</td></tr><tr><td>5</td><td>15.6+0.2</td><td>20.8+0.1</td><td>16.3+4.6</td><td></td><td>18.9+2.3</td></tr><tr><td>6</td><td>19.6+0.2</td><td>19.0+0.2</td><td>14.9+5.4</td><td></td><td>18.3+2.6</td></tr><tr><td>7</td><td>20.1+0.2</td><td>19.4+0.1</td><td>14.7+4.9</td><td></td><td>16.1+2.4</td></tr><tr><td>8</td><td>19.6+0.2</td><td>18.4+0.1</td><td>16.7+5.3</td><td></td><td>15.0+2.0</td></tr><tr><td>9</td><td>22.3+0.2</td><td>20.3+0.2</td><td>14.8+5.2</td><td></td><td>21.9+1.7</td></tr><tr><td>10</td><td>19.7+0.2</td><td>22.3+0.1</td><td>11.7+5.5</td><td></td><td>15.2+2.0</td></tr><tr><td><strong>Total</strong></td><td><strong>224.8</td></strong><td><strong>200.7</td></strong><td><strong>215.1</td></strong><td><strong></td></strong><td><strong>200.5</td></strong></tr><tr><td rowspan="12">HARRIS</td></tr><tr><td>1</td><td>24.8+0.0</td><td>16.5+0.1</td><td>15.8+4.8</td><td></td><td>14.8+2.0</td></tr><tr><td>2</td><td>12.7+0.0</td><td>15.3+0.1</td><td>15.5+4.8</td><td></td><td>13.6+1.4</td></tr><tr><td>3</td><td>17.3+0.1</td><td>17.1+0.1</td><td>12.0+4.6</td><td></td><td>17.6+1.7</td></tr><tr><td>4</td><td>17.8+0.0</td><td>15.4+0.1</td><td>16.6+4.6</td><td></td><td>19.4+2.4</td></tr><tr><td>5</td><td>15.2+0.1</td><td>18.8+0.1</td><td>16.2+5.2</td><td></td><td>21.5+2.2</td></tr><tr><td>6</td><td>41.5+0.1</td><td>45.1+0.1</td><td>33.6+5.3</td><td></td><td>49.4+2.5</td></tr><tr><td>7</td><td>17.2+0.0</td><td>15.9+0.1</td><td>14.1+4.4</td><td></td><td>13.0+2.1</td></tr><tr><td>8</td><td>21.1+0.1</td><td>19.7+0.1</td><td>21.0+5.1</td><td></td><td>16.1+1.5</td></tr><tr><td>9</td><td>16.6+0.0</td><td>18.7+0.1</td><td>12.9+4.2</td><td></td><td>18.8+1.8</td></tr><tr><td>10</td><td>21.1+0.0</td><td>24.4+0.1</td><td>22.8+5.3</td><td></td><td>17.8+1.7</td></tr><tr><td><strong>Total</strong></td><td><strong>205.9</td></strong><td><strong>208.2</td></strong><td><strong>228.5</td></strong><td><strong></td></strong><td><strong>221.3</td></strong></tr><tr><td rowspan="12">FAST</td></tr><tr><td>1</td><td>1.1+0.2</td><td>1.3+0.2</td><td>0.8+5.4</td><td></td><td>0.9+2.7</td></tr><tr><td>2</td><td>1.3+0.2</td><td>1.4+0.2</td><td>1.3+5.8</td><td></td><td>1.3+2.6</td></tr><tr><td>3</td><td>1.3+0.1</td><td>1.2+0.1</td><td>1.0+4.4</td><td></td><td>1.4+3.0</td></tr><tr><td>4</td><td>1.3+0.1</td><td>1.3+0.1</td><td>1.1+5.6</td><td></td><td>1.2+2.9</td></tr><tr><td>5</td><td>1.2+0.1</td><td>0.9+0.2</td><td>1.2+5.7</td><td></td><td>1.3+2.7</td></tr><tr><td>6</td><td>1.2+0.1</td><td>1.1+0.1</td><td>1.3+5.2</td><td></td><td>1.0+2.6</td></tr><tr><td>7</td><td>0.9+0.1</td><td>0.9+0.1</td><td>1.4+5.5</td><td></td><td>1.2+2.4</td></tr><tr><td>8</td><td>0.9+0.1</td><td>1.1+0.1</td><td>1.2+5.4</td><td></td><td>1.5+2.8</td></tr><tr><td>9</td><td>1.1+0.1</td><td>1.1+0.1</td><td>0.9+4.6</td><td></td><td>1.3+2.7</td></tr><tr><td>10</td><td>1.1+0.1</td><td>0.9+0.1</td><td>1.3+5.2</td><td></td><td>1.3+2.8</td></tr><tr><td><strong>Total</strong></td><td><strong>12.8</td></strong><td><strong>12.5</td></strong><td><strong>64.4</td></strong><td><strong></td></strong><td><strong>39.7</td></strong></tr><tr><td rowspan="12">BRISK</td></tr><tr><td>1</td><td>30.5+0.1</td><td>22.7+0.5</td><td>22.3+6.2</td><td></td><td>22.8+4.0</td></tr><tr><td>2</td><td>22.1+0.1</td><td>22.3+0.5</td><td>22.2+5.5</td><td></td><td>28.2+3.9</td></tr><tr><td>3</td><td>22.9+0.1</td><td>25.4+0.6</td><td>22.5+5.5</td><td></td><td>21.9+4.3</td></tr><tr><td>4</td><td>22.2+0.1</td><td>22.3+0.5</td><td>21.6+5.2</td><td></td><td>22.1+4.1</td></tr><tr><td>5</td><td>21.9+0.1</td><td>21.4+0.5</td><td>22.1+4.3</td><td></td><td>22.5+3.5</td></tr><tr><td>6</td><td>21.9+0.1</td><td>21.9+0.5</td><td>21.9+5.5</td><td></td><td>20.6+3.9</td></tr><tr><td>7</td><td>21.5+0.1</td><td>23.4+0.5</td><td>22.0+4.6</td><td></td><td>21.5+3.9</td></tr><tr><td>8</td><td>21.7+0.1</td><td>21.5+0.5</td><td>21.8+5.6</td><td></td><td>21.7+4.3</td></tr><tr><td>9</td><td>21.1+0.1</td><td>21.5+0.6</td><td>21.4+5.3</td><td></td><td>21.5+4.2</td></tr><tr><td>10</td><td>22.0+0.1</td><td>21.3+0.6</td><td>22.5+5.7</td><td></td><td>22.8+3.3</td></tr><tr><td><strong>Total</strong></td><td><strong>228.9</td></strong><td><strong>229.0</td></strong><td><strong>273.8</td></strong><td><strong></td></strong><td><strong>265.1</td></strong></tr><tr><td rowspan="12">ORB</td></tr><tr><td>1</td><td>9.4+0.0</td><td>9.9+0.6</td><td>7.0+4.7</td><td></td><td>9.9+5.0</td></tr><tr><td>2</td><td>7.9+0.1</td><td>9.0+0.5</td><td>7.9+4.8</td><td></td><td>8.8+4.6</td></tr><tr><td>3</td><td>7.8+0.1</td><td>10.2+0.7</td><td>8.9+5.1</td><td></td><td>7.3+5.8</td></tr><tr><td>4</td><td>8.0+0.1</td><td>6.7+0.4</td><td>9.1+4.9</td><td></td><td>8.4+5.2</td></tr><tr><td>5</td><td>9.5+0.1</td><td>6.9+0.4</td><td>10.1+5.1</td><td></td><td>10.3+5.6</td></tr><tr><td>6</td><td>9.4+0.1</td><td>7.5+0.6</td><td>9.1+4.9</td><td></td><td>13.3+4.8</td></tr><tr><td>7</td><td>9.4+0.1</td><td>9.2+0.6</td><td>7.7+5.4</td><td></td><td>9.6+5.4</td></tr><tr><td>8</td><td>9.6+0.1</td><td>8.4+0.7</td><td>8.6+5.5</td><td></td><td>7.5+5.8</td></tr><tr><td>9</td><td>7.1+0.1</td><td>7.6+0.6</td><td>10.1+4.7</td><td></td><td>9.9+6.3</td></tr><tr><td>10</td><td>9.2+0.1</td><td>9.1+0.6</td><td>9.0+5.4</td><td></td><td>12.4+5.4</td></tr><tr><td><strong>Total</strong></td><td><strong>88.3</td></strong><td><strong>90.3</td></strong><td><strong>137.9</td></strong><td><strong></td></strong><td><strong>151.4</td></strong></tr><tr><td rowspan="12">AKAZE</td></tr><tr><td>1</td><td>108.4+0.1</td><td>82.2+0.3</td><td>90.3+5.7</td><td>82.7+7.7</td><td>79.9+3.3</td></tr><tr><td>2</td><td>81.2+0.1</td><td>81.3+0.4</td><td>85.2+4.9</td><td>94.8+8.4</td><td>87.5+3.7</td></tr><tr><td>3</td><td>81.9+0.1</td><td>86.8+0.4</td><td>78.1+4.6</td><td>87.4+8.3</td><td>103.5+3.6</td></tr><tr><td>4</td><td>92.5+0.2</td><td>86.3+0.5</td><td>87.2+5.8</td><td>83.7+7.5</td><td>85.7+3.3</td></tr><tr><td>5</td><td>81.3+0.2</td><td>87.5+0.4</td><td>85.5+5.6</td><td>89.8+8.3</td><td>83.2+2.7</td></tr><tr><td>6</td><td>85.4+0.1</td><td>85.0+0.3</td><td>106.2+5.4</td><td>93.0+7.9</td><td>80.7+2.7</td></tr><tr><td>7</td><td>82.7+0.2</td><td>86.9+0.4</td><td>84.7+5.2</td><td>88.9+8.1</td><td>85.9+2.9</td></tr><tr><td>8</td><td>77.0+0.1</td><td>92.4+0.5</td><td>83.4+5.0</td><td>88.7+7.8</td><td>86.7+2.6</td></tr><tr><td>9</td><td>85.0+0.1</td><td>102.6+0.4</td><td>85.5+4.7</td><td>92.0+7.9</td><td>79.5+3.5</td></tr><tr><td>10</td><td>86.9+0.1</td><td>86.7+0.4</td><td>84.1+5.7</td><td>84.8+7.6</td><td>88.1+3.3</td></tr><tr><td><strong>Total</strong></td><td><strong>863.7</td></strong><td><strong>881.7</td></strong><td><strong>923.0</td></strong><td><strong>965.3</td></strong><td><strong>892.2</td></strong></tr><tr><td rowspan="12">SIFT</td></tr><tr><td>1</td><td>156.2+0.1</td><td></td><td>171.5+5.2</td><td></td><td>164.3+10.7</td></tr><tr><td>2</td><td>168.5+0.1</td><td></td><td>150.1+5.4</td><td></td><td>115.6+12.6</td></tr><tr><td>3</td><td>156.5+0.1</td><td></td><td>148.5+5.4</td><td></td><td>146.2+10.7</td></tr><tr><td>4</td><td>163.9+0.1</td><td></td><td>167.9+5.9</td><td></td><td>151.9+11.1</td></tr><tr><td>5</td><td>204.7+0.1</td><td></td><td>198.6+4.8</td><td></td><td>148.6+11.1</td></tr><tr><td>6</td><td>170.0+0.1</td><td></td><td>173.2+5.2</td><td></td><td>145.6+9.4</td></tr><tr><td>7</td><td>176.1+0.1</td><td></td><td>144.5+4.6</td><td></td><td>148.0+11.1</td></tr><tr><td>8</td><td>181.0+0.1</td><td></td><td>155.7+5.4</td><td></td><td>144.4+11.9</td></tr><tr><td>9</td><td>163.3+0.1</td><td></td><td>157.9+5.4</td><td></td><td>144.5+12.1</td></tr><tr><td>10</td><td>185.0+0.1</td><td></td><td>181.4+5.2</td><td></td><td>146.3+8.9</td></tr><tr><td><strong>Total</strong></td><td><strong>1726.3</td></strong><td><strong></td></strong><td><strong>1701.7</td></strong><td><strong></td></strong><td><strong>1564.9</td></strong></tr></table>

I understand you're asking for the three best combinations with the sole criterion of speed, so I've created a performance table that shows the number of matching points per millisecond.

<table><tr><th></th><th>BRIEF</th><th>ORB</th><th>FREAK</th><th>AKAZE</th><th>SIFT</th></tr><tr><td>SHITOMASI</td><td>4.2</td><td>4.5</td><td>3.6</td><td></td><td>4.6</td></tr><tr><td>HARRIS</td><td>0.8</td><td>0.8</td><td>0.6</td><td></td><td>0.7</td></tr><tr><td>FAST</td><td>86.1</td><td>86.0</td><td>13.6</td><td></td><td>26.3</td></tr><tr><td>BRISK</td><td>4.3</td><td>4.0</td><td>3.4</td><td></td><td>3.7</td></tr><tr><td>ORB</td><td>5.1</td><td>7.4</td><td>2.9</td><td></td><td>4.6</td></tr><tr><td>AKAZE</td><td>1.5</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.4</td></tr><tr><td>SIFT</td><td>0.4</td><td></td><td>0.3</td><td></td><td>0.5</td></tr></table>

So the best three combinations are: __FAST/BRIEF__, __FAST/ORB__, __FAST/SIFT__.
