// Includes
#include "opencv2/opencv.hpp"
#include <vector>
#include <fstream>
#include <math.h>
#include <chrono>
#include <thread>

using namespace std::chrono;
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
Mat pre_process(Mat img, Mat homography, bool save=false, string imageName="empty.jpg"){
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

Mat getPart(Mat frame, int part, int num, int mode = 0){
	int height = frame.size().height;
	int width = frame.size().width;
	if(mode == 0){
		if(num != part) return frame(Rect(0, height*(part-1)/num, width, height/num)).clone();
		else return frame(Rect(0, height*(part-1)/num, width, height-height*(part-1)/num)).clone();
	}
	if(num != part) return frame(Rect(width*(part-1)/num, 0, width/num, height)).clone();
	else return frame(Rect(width*(part-1)/num, 0, width-width*(part-1)/num, height)).clone();
}

pair<vector<vector<double>>, int> getDensityDataSpatial(VideoCapture &cap, Mat &frame_empty, int step = 3, int part = 1, int num = 1, int th = 0, int mode = 0){
	// Find homography using a default set of points
	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0; // Average densities over 15 (fps) frames per second
	Mat frame_old_difference; // Old frame to get optical flow

	frame_empty = pre_process(frame_empty, default_homography); // Apply transformation and crop to background
	
	int original_size = frame_empty.size().height * frame_empty.size().width;
	frame_empty = getPart(frame_empty, part, num, mode);

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	
	vector<vector<double>> result;
	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current); // read a new frame from video
		// Exit if no more frames available
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography); // Apply transformation and crop
		frame_processed = getPart(frame_processed, part, num, mode);

		absdiff(frame_processed, frame_empty, frame_difference); // Background subtraction
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); // Smoothen and fill gaps

		// Set the old frame to the current frame in case of frame 0
		if(frame_count == 0) frame_old_difference = frame_difference;

		// flow is an image where all moving pixels are white and all stationary pixels are black
		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); // Smoothen and fill gaps

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		total_density += pixel_ratio;
		dynamic_density += min(dynamic_pixel_ratio, pixel_ratio);

		if(frame_count % step == 0 and frame_count != 0){

			// Every step, evaluate average densities
			dynamic_density /= step;
			total_density /= step;
			result.push_back({(double)frame_count, total_density*size, min(dynamic_density, 0.95 * total_density)*size});
			

			total_density = 0;
			dynamic_density = 0;
		}

		frame_count++; // Update frame count

		frame_old_difference = frame_difference; // Update old frame
	}
	return {result, original_size};
}
pair<double, double> getAccuracy(vector<vector<double>> actual, vector<vector<double>> baseline, string metric = "square"){
	if(baseline.size() != actual.size()){
		return {-1, -1};
	}
	pair<double, double> result;
	for(int i=0;i<baseline.size();i++){
		if(metric == "abs"){
			result.first += abs(baseline[i][1]-actual[i][1]);
			result.second += abs(baseline[i][2]-actual[i][2]);
		}
		else{
			result.first += (baseline[i][1]-actual[i][1]) * (baseline[i][1]-actual[i][1]);
			result.second += (baseline[i][2]-actual[i][2]) * (baseline[i][2]-actual[i][2]);
		}
	}
	result.first /= baseline.size();
	result.second /= baseline.size();
	// cout<< result.first <<" "<<result.second<<endl;
	return {(double)round(10000*(1-tanh(result.first)))/100, (double)round(10000*(1-tanh(result.second)))/100};
}
vector<vector<double>> load_file(string file_name = "baseline.txt"){
	fstream f("Outputs/"+file_name);
	vector<vector<double>> result;
	string line, val;
	getline(f, line);
	while(getline(f, line)){
		stringstream s(line);
		vector<double> temp;
		while(getline(s, val, ',')){
			temp.push_back(stod(val));
		}
		if(temp.size() == 3){
			result.push_back(temp);
		}
	}
	f.close();
	return result;
}

vector<vector<double>> getDensityDataTemporal(VideoCapture &cap, Mat &frame_empty, int step = 3, int part = 1, int num = 1, int th = 0){
	
	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	vector<vector<double>> result;

	int frames = cap.get(CAP_PROP_FRAME_COUNT);

	int start = (frames/num)*(part-1);
	int actual_start = start + (step - (start % step)) % step;
	if(actual_start >= frames) return result;
	actual_start = actual_start - step;
	actual_start = max(actual_start, 0);
	int dur = frames/num;
	if(part == num) dur += frames % num;
	dur += start - actual_start;

	// cout << dur << " " << start << " " << actual_start << endl;

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0; // Average densities over 15 (fps) frames per second
	Mat frame_old_difference; // Old frame to get optical flow

	frame_empty = pre_process(frame_empty, default_homography); // Apply transformation and crop to background

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	// cap.set(CAP_PROP_POS_FRAMES, actual_start);
	Mat frame_current;
	int ct = 0;
	while(ct++ < actual_start){
		cap.read(frame_current);
	}
	while(true){
		if(frame_count == dur) break;
		Mat frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current); // read a new frame from video
		// Exit if no more frames available
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography); // Apply transformation and crop

		absdiff(frame_processed, frame_empty, frame_difference); // Background subtraction
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); // Smoothen and fill gaps

		// Set the old frame to the current frame in case of frame 0
		if(frame_count == 0) frame_old_difference = frame_difference;

		// flow is an image where all moving pixels are white and all stationary pixels are black
		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); // Smoothen and fill gaps

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density
		// imshow("Queue Density", flow);
		if(frame_count != 0){
			total_density += pixel_ratio;
			dynamic_density += min(dynamic_pixel_ratio, pixel_ratio);
		}
		else{
			total_density = 0;
			dynamic_density = 0;
		}

		if((frame_count + actual_start) % step == 0 and frame_count != 0){
			dynamic_density /= step;
			total_density /= step;
			result.push_back({(double)actual_start + frame_count, total_density, min(dynamic_density, 0.95 * total_density)});

			total_density = 0;
			dynamic_density = 0;
		}

		frame_count++; // Update frame count

		frame_old_difference = frame_difference; // Update old frame
	}
	return result;
}