// Includes
#include "opencv2/opencv.hpp"
#include <vector>

using namespace cv;
using namespace std;

// Constants

// Points given by ma'am in the specifications
const vector<Point2f> GIVEN_POINTS = {Point2f(472,52), Point2f(472,830), Point2f(800,830), Point2f(800,52)};

// (x1, y1, width, height) of the rectangle formed by ma'am's points 
const Rect RECT_CROP(GIVEN_POINTS[0].x, GIVEN_POINTS[0].y, GIVEN_POINTS[2].x - GIVEN_POINTS[0].x, GIVEN_POINTS[1].y - GIVEN_POINTS[0].y);

// Default set of points selected in Part 1
// The user is not expected to manually select points everytime when running the program
const vector<Point2f> DEFAULT_POINTS = {Point2f(979, 196), Point2f(393, 1058), Point2f(1540, 1055), Point2f(1272, 202)};

// Default homography matrix
Mat default_homography;

// Functions

// Convert image to grayscale
Mat grayScale(Mat & img){
	Mat im_gray;
	cvtColor(img, im_gray, COLOR_BGR2GRAY);
	return im_gray;
}

// Transform and return cropeed image without any manual input from user
Mat pre_process(Mat & img, Mat & homography, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop, im_gray = grayScale(img);
	warpPerspective(im_gray, im_transform, homography, im_gray.size());
	im_crop = im_transform(RECT_CROP);
	if(save){
		imwrite("Crops/crop_" + imageName, im_crop);
	}
	return im_crop;
}

// Smoothen and fill gaps
void filter(Mat & img){
	Mat result;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));
	morphologyEx(img, result, MORPH_CLOSE, kernel, Point(-1,-1), 1);
	morphologyEx(result, result, MORPH_OPEN, kernel, Point(-1, -1), 1);
	dilate(result, result, kernel, Point(-1,-1), 1);
	dilate(result, result, kernel, Point(-1,-1), 1);
	img = result;
}

// Get optical flow
Mat getFlow(Mat & frame_old_difference, Mat & frame_difference){
	Mat mag, theta, vx_vy[2], temp[3], temp_merge;
	Mat flow(frame_old_difference.size(), CV_32FC2);
	
	// Calculate optical flow
	calcOpticalFlowFarneback(frame_old_difference, frame_difference, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	// Split flow in x and y components
	split(flow, vx_vy);

	// Convert x and y components of flow to polar co-ordinates
	cartToPolar(vx_vy[0], vx_vy[1], mag, theta, true);

	// Normalize to get values of magnitude between 0 and 1
	normalize(mag, mag, 0.0f, 1.0f, NORM_MINMAX);

	theta *= (double)1 / 510;

	temp[0] = theta;
	temp[1] = Mat::ones(theta.size(), CV_32F);
	temp[2] = mag;

	merge(temp, 3, temp_merge);
	temp_merge.convertTo(temp_merge, CV_8U, 255.0);
	cvtColor(temp_merge, temp_merge, COLOR_HSV2BGR);
	temp_merge = grayScale(temp_merge);

	// temp_merge distinctly shows moving pixels
	return temp_merge;
}