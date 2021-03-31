//========================================== =============================================


//========================================== Includes =============================================
#include "opencv2/opencv.hpp"
#include <vector>
#include <fstream>
#include <math.h>
#include <chrono>

using namespace std::chrono;
using namespace cv;
using namespace std;

//========================================== Constants =============================================


// Points given by ma'am in the specifications
const vector<Point2f> GIVEN_POINTS = {Point2f(472,52), Point2f(472,830), Point2f(800,830), Point2f(800,52)};

// (x1, y1, width, height) of the rectangle formed by ma'am's points 
const Rect RECT_CROP(GIVEN_POINTS[0].x, GIVEN_POINTS[0].y, GIVEN_POINTS[2].x - GIVEN_POINTS[0].x, GIVEN_POINTS[1].y - GIVEN_POINTS[0].y);

// Default set of points selected in Part 1
// The user is not expected to manually select points everytime when running the program
const vector<Point2f> DEFAULT_POINTS = {Point2f(979, 196), Point2f(393, 1058), Point2f(1540, 1055), Point2f(1272, 202)};

// Default homography matrix
Mat default_homography;

// constants used when resolution of image is chamged 
vector<Point2f> GIVEN_POINTS_RESOLUTION;
Rect RECT_CROP_RESOLUTION;
vector<Point2f> DEFAULT_POINTS_RESOLUTION;



//========================================== Functions =============================================

// auxilary printing function
void print_result(vector<vector<double>> &result, int i ) {
	cout << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
}


// Convert image to grayscale
Mat grayScale(Mat & img){
	Mat im_gray;
	cvtColor(img, im_gray, COLOR_BGR2GRAY);
	return im_gray;
}

// Transform and return croped grayscaled image without any manual input from user
Mat pre_process(Mat img, Mat homography, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop, im_gray = grayScale(img);
	warpPerspective(im_gray, im_transform, homography, im_gray.size());
	im_crop = im_transform(RECT_CROP);
	return im_crop;
}
// overloading previous function when processing a different resolution 
Mat pre_process(Mat img, Mat homography, Rect RECT_CROP, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop, im_gray = grayScale(img);
	warpPerspective(im_gray, im_transform, homography, im_gray.size());
	im_crop = im_transform(RECT_CROP);
	return im_crop;
}

// Transform and return image without any manual input from user (without grayscaling)
Mat pre_process_color(Mat img, Mat homography, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop;
	warpPerspective(img, im_transform, homography, img.size());
	im_crop = im_transform(RECT_CROP);
	return im_crop;
}


// scale a vector by a given factor
vector<Point2f> scaling (const vector<Point2f> &a, double fx, double fy) {
	vector<Point2f> b;
	for (int i = 0; i < a.size() ; ++i) {
		b.push_back(Point2f(round (a[i].x * fx) , round (a[i].y * fy) ) );
	}
	return b;
}
// obtain a rectange from 4 points
Rect getRect (const vector<Point2f> &a) {
	Rect RECT_CROP(a[0].x, a[0].y, a[2].x - a[0].x, a[1].y - a[0].y);
	return RECT_CROP;
}
// print a vector of points
void print_points (vector<Point2f> &a) {
	for (int i = 0; i < a.size() ; ++i) {
		cout << "( " << a[i].x << ", " << a[i].y << ")" << "	";
	}
	cout << endl;
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

// For example, if part == 2, num == 5, this returns the second rectangle if an image is divided into 5 rectangles
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


// Function to check if given string is an integer
int isint(string var) {

    if (var.size() > 7 or var.size() < 1)
        return 0;

    if (var[1] == 'x' || var[2] == 'x')
    {
        int idx = 0;
        if (var[0] == '-')
        {
            idx = 1;
        }
        if (var[idx] != '0')
            return 0;
        for (int i = idx + 2; i < var.size(); i++)
        {
            if (!isxdigit(var[i]))
                return 0;
        }
        int d = stoi(var, 0, 16);
        if (d < -(1 << 15) or d > (1 << 15) - 1)
            return 0;
        return 16;
    }
    else
    {
        if (!isdigit(var[0]) and var[0] != '-')
            return 0;
        for (int i = 1; i < var.size(); i++)
        {
            if (!isdigit(var[i]))
                return 0;
        }
        int d = stoi(var);
        if (d < -(1 << 15) or d > (1 << 15) - 1)
            return 0;
        return 10;
    }
}

//========================================== Optial Flow Functions =============================================


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


// get Sparse Optical Flow Using Lucas Kanade Algorithm
Mat getSparseFlow(Mat & frame_old_difference, Mat & frame_difference, Mat & temp){
	
	Mat img;

	// Create random colors
	vector<Scalar> colors_rand;
    RNG wheel;
    for(int i = 0; i < 200; i++)
    {
        int red = wheel.uniform(0, 256);
        int green = wheel.uniform(0, 256);
        int blue = wheel.uniform(0, 256);
        colors_rand.push_back(Scalar(red ,green ,blue));
    }

    vector<Point2f> point0, point1;

    // find points of interest
    goodFeaturesToTrack(frame_difference, point0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // in case there are no interesting points
    if (point0.size() == 0) return grayScale(temp);

    // Create a mask image for drawing purposes
	Mat mask_img = Mat::zeros(temp.size(), temp.type());

	// Calculate optical flow
	vector<uchar> ok;
    vector<float> bug;
    TermCriteria c = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(frame_old_difference, frame_difference, point0, point1, ok, bug, Size(15,15), 2, c);
    vector<Point2f> points_select;

    for(uint i = 0; i < point0.size(); i++) {
        // Select good points
        if(ok[i] == 1) {
            points_select.push_back(point1[i]);
            // draw on image
            line(mask_img,point1[i], point0[i], colors_rand[i], 2);
            circle(temp, point1[i], 5, colors_rand[i], -1);
        }
    }
    
    add(temp, mask_img, img);
    img = grayScale(img);
    return img;
}


//========================================== Baseline =============================================


vector<vector<double>> getDensityData(VideoCapture &cap) {

	// store densities in form: row[i] = {frame_count, queue density, dynamic density}
	vector<vector<double>> result;  

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	Mat frame_old_difference;

	// get and process background image
	Mat frame_empty = imread("../Data/Images/empty.jpg");
	frame_empty = pre_process(frame_empty, default_homography);

	// compute size of all frames is same
	int size = frame_empty.size().height * frame_empty.size().width;


	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);

		// reached end of video no more frame available
		if(success == false) break;

		frame_processed = pre_process(frame_current, default_homography);

		absdiff(frame_processed, frame_empty, frame_difference); // Background Subtraction
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); //smoothen and fill gaps

		if(frame_count == 0) frame_old_difference = frame_difference;

		// get flow from current frame and prev fram
		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); 

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get queue density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		total_density = pixel_ratio;
		dynamic_density = min(dynamic_pixel_ratio, pixel_ratio);
		dynamic_density = min(dynamic_density, 0.95 * total_density);
		
		result.push_back( { (double)frame_count, total_density, dynamic_density } );

		// print_result(result, frame_count);	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}

//========================================== Method 1 =============================================

// process frame at intervals of X frames. ie process frame N and then N+X.
vector<vector<double>> getDensityDataSkips (VideoCapture &cap, int X = 1) {

	// store densities in form: row[i] = {frame_count, queue density, dynamic density}
	vector<vector<double>> result; 

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS); 

	// get and process background image
	Mat frame_empty = imread("../Data/Images/empty.jpg");
	frame_empty = pre_process(frame_empty, default_homography);

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	
	Mat frame_prev; //prev frame needs to be stored
	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	while(true) {

		Mat frame_current; 
		Mat frame_processed;
		Mat frame_prev_processed; 
		Mat frame_threshold;
		Mat frame_difference; // need to be local or clear it
		Mat frame_prev_difference; // need to be local or clear it 

		// reached end of video no more frame available
		bool success = cap.read(frame_current);
		if (success == false) break;

		// if frame_count is not a multiple of X then don't process it, it's result is same as that of previous frame.
		if (frame_count % X != 0) {
			result.push_back( { (double) frame_count, result[frame_count - 1][1], result[frame_count - 1][2] });

			// print_result(result, frame_count);	
			
			frame_count++;
			frame_prev = frame_current;
			continue;
		}

		frame_processed = pre_process(frame_current, default_homography); 
		absdiff(frame_processed, frame_empty, frame_difference); //Baackground subtraction

		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); 

		if(frame_count == 0) {
			frame_prev = frame_current;
		}

		frame_prev_processed = pre_process(frame_prev, default_homography); //process prev frame
		absdiff(frame_prev_processed, frame_empty, frame_prev_difference); 

		Mat flow = getFlow(frame_prev_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow);

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		total_density = pixel_ratio;
		dynamic_density = min(dynamic_pixel_ratio, pixel_ratio);
		dynamic_density = min(dynamic_density, 0.95 * total_density);

		result.push_back( { (double)frame_count, total_density, dynamic_density } );
		
		// print_result(result, frame_count);	

		frame_count++;
		frame_prev = frame_current;
	}
	return result;
}

//========================================== Method 2 =============================================

// function to process image at specified X, Y resolutions.
// resolution of frame_processed (cropped image after homography) is changed 
vector<vector<double>> getDensityDataResolutionEasy(VideoCapture &cap, int X, int Y) {

	
	// X is width in number of pixels
	// Y is height in number of pixels
	Size dim(X, Y); 

	vector<vector<double>> result;

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	Mat frame_old_difference;
	Mat frame_empty = imread("../Data/Images/empty.jpg");
	frame_empty = pre_process(frame_empty, default_homography);

	resize(frame_empty, frame_empty, dim);

	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);

		if(success == false){
			break;
		}

		frame_processed = pre_process(frame_current, default_homography);
		
		resize(frame_processed, frame_processed, dim); //resize the processed frame as per given resolution

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); 

		if(frame_count == 0) frame_old_difference = frame_difference;

		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); 

		int size = frame_processed.size().height * frame_processed.size().width; //size of frame
		double pixel_ratio = (double) countNonZero(frame_threshold) / size; //queue density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; //dynamic density

		total_density = pixel_ratio;
		dynamic_density = min(dynamic_pixel_ratio, pixel_ratio);
		dynamic_density = min(dynamic_density, 0.95 * total_density);
		
		result.push_back( { (double)frame_count, total_density, dynamic_density } );

		// print_result(result, frame_count);	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}


// function to process image at specified X, Y resolutions.
// resolution of original image is changed
vector<vector<double>> getDensityDataResolution (VideoCapture &cap, int X, int Y, int original_X, int original_Y ) {

	Size dim(X, Y);

	// obtain scaling factor
	double fx = (double)X / (double)original_X;
	double fy = (double)Y / (double)original_Y;

	// scale the constants needed for homography
	GIVEN_POINTS_RESOLUTION = scaling(GIVEN_POINTS, fx, fy);
	RECT_CROP_RESOLUTION = getRect (GIVEN_POINTS_RESOLUTION);
	DEFAULT_POINTS_RESOLUTION = scaling(DEFAULT_POINTS, fx, fy);

	// print_points(GIVEN_POINTS_RESOLUTION);
	// print_points(DEFAULT_POINTS_RESOLUTION);


	vector<vector<double>> result;

	// homography as per resolution
	default_homography = findHomography(DEFAULT_POINTS_RESOLUTION, GIVEN_POINTS_RESOLUTION);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	Mat frame_old_difference;
	Mat frame_empty = imread("../Data/Images/empty.jpg");

	resize(frame_empty, frame_empty, dim); //resolve

	frame_empty = pre_process(frame_empty, default_homography, RECT_CROP_RESOLUTION);


	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);

		if(success == false){
			break;
		}
	
		resize(frame_current, frame_current, dim); // resolve original frame image and then process it

		frame_processed = pre_process(frame_current, default_homography, RECT_CROP_RESOLUTION);

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); 

		if(frame_count == 0) frame_old_difference = frame_difference;

		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); 

		int size = frame_processed.size().height * frame_processed.size().width; 
		double pixel_ratio = (double) countNonZero(frame_threshold) / size;
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size;

		total_density = pixel_ratio;
		dynamic_density = min(dynamic_pixel_ratio, pixel_ratio);
		dynamic_density = min(dynamic_density, 0.95 * total_density);
		
		result.push_back( { (double)frame_count, total_density, dynamic_density } );
	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}

//========================================== Method 3 =============================================


// Function to process various parts of an image spatially (Method 3)
pair<vector<vector<double>>, int> getDensityDataSpatial(VideoCapture &cap, Mat &frame_empty, int step = 1, int part = 1, int num = 1, int mode = 0){
	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;
	Mat frame_old_difference;

	frame_empty = pre_process(frame_empty, default_homography);
	
	int original_size = frame_empty.size().height * frame_empty.size().width;
	frame_empty = getPart(frame_empty, part, num, mode);

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	vector<vector<double>> result; // result stored as a vector of 3-D vectors
	
	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);
		
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography);
		
		frame_processed = getPart(frame_processed, part, num, mode); // Get a rectangular region of the frame

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold);

		if(frame_count == 0) frame_old_difference = frame_difference;

		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow);

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		if(frame_count != 0){
			total_density += pixel_ratio;
			dynamic_density += min(dynamic_pixel_ratio, pixel_ratio);
		}
		else{
			total_density = 0;
			dynamic_density = 0;
		}

		// Skip every step number of frames
		if(frame_count % step == 0 and frame_count != 0){
			dynamic_density /= step;
			total_density /= step;

			// Add frame data to result vector
			result.push_back( { (double)frame_count, total_density * size, min(dynamic_density, 0.95 * total_density) * size} );
			

			total_density = 0;
			dynamic_density = 0;
		}

		frame_count++;

		frame_old_difference = frame_difference;
	}
	// Return the frame data, with the original size of cropped image
	// Size is required to calculate pixel density
	return {result, original_size};
}

//========================================== Method 4 =============================================

// Function to process various parts of a video temporally (Method 4)
vector<vector<double>> getDensityDataTemporal(VideoCapture &cap, Mat &frame_empty, int step = 1, int part = 1, int num = 1){
	
	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	vector<vector<double>> result; // Vector to store the result

	int frames = cap.get(CAP_PROP_FRAME_COUNT);

	int start = (frames/num)*(part-1);
	int actual_start = start + (step - (start % step)) % step;
	if(actual_start >= frames) return result;
	actual_start = actual_start - step;
	actual_start = max(actual_start, 0);
	int dur = frames/num;
	if(part == num) dur += frames % num;
	dur += start - actual_start;

	// actual_start gives the frame number to start processing
	// dur gives the total duration to process for
	// This is done to handle all corner cases when frames are divided temporally to make sure result is complete

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;
	Mat frame_old_difference;

	frame_empty = pre_process(frame_empty, default_homography); // Apply transformation and crop to background

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	Mat frame_current;
	int ct = 0;

	// Seek the frame number actual_start to begin processing
	while(ct++ < actual_start){
		cap.read(frame_current);
	}

	// Begin processing from actual_start
	while(true){

		if(frame_count == dur) break; // Exit if the number of frames processed exceeds the duration for this thread

		Mat frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography);

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold);

		if(frame_count == 0) frame_old_difference = frame_difference;

		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow);

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density
		
		if(frame_count != 0){
			total_density += pixel_ratio;
			dynamic_density += min(dynamic_pixel_ratio, pixel_ratio);
		}
		else{
			total_density = 0;
			dynamic_density = 0;
		}

		// Skip every step number of frames
		if((frame_count + actual_start) % step == 0 and frame_count != 0){
			dynamic_density /= step;
			total_density /= step;

			// Add frame data to result vector
			result.push_back( { (double) actual_start + frame_count, total_density, min(dynamic_density, 0.95 * total_density) } );

			total_density = 0;
			dynamic_density = 0;
		}

		frame_count++;

		frame_old_difference = frame_difference;
	}
	return result;
}

//========================================== Bonus (Sparse Optical Flow) =============================================

// Function to process video with sparse optical flow (Bonus)
vector<vector<double>> getDensityDataSparse(string file_name, Mat &frame_empty, int step = 1){
	
	VideoCapture cap;
	cap.open("../Data/Videos/"+file_name); // Open video file

	// if not success, exit program
    if (cap.isOpened() == false) {
        cout << "Cannot open the video file. Please provide a valid name (refer to README.md)." << endl;
        exit(3);
    }

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	vector<vector<double>> result; // Vector to store the result

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;
	Mat frame_old_difference;

	frame_empty = pre_process(frame_empty, default_homography); // Apply transformation and crop to background

	int size = frame_empty.size().height * frame_empty.size().width; // Size of frame

	Mat frame0, frame0_difference;

	// Begin processing from actual_start
	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography);

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold);

		if(frame_count == 0){ 
			frame_old_difference = frame_difference; 
			frame0 = frame_current;
			frame0_difference = frame_difference;
		}

		Mat temp = pre_process_color(frame0, default_homography);

		Mat flow = getSparseFlow(frame_old_difference, frame_difference, temp);
		threshold(flow, flow, 230, 255.0, THRESH_BINARY);
		// filter(flow);

		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		total_density = pixel_ratio;
		dynamic_density = min(dynamic_pixel_ratio, pixel_ratio);
		dynamic_density = min(dynamic_density, 0.95 * total_density);

		result.push_back( { (double)frame_count, total_density, dynamic_density } );

		frame_count++;

		frame_old_difference = frame_difference;
	}
	cap.release(); // Close the VideoCapture
	return result;
}

//========================================== Helper Functions for Analysis and accuracy =============================================

// Given a baseline, this calculates the accuracy of Queue density and Dynamic density for a result
pair<double, double> getAccuracy(vector<vector<double>> actual, vector<vector<double>> baseline, string metric = "rms"){
	
	// The metric can be average absolute error or average square error

	if(baseline.size() != actual.size()){
		return {-1, -1};
	}
	pair<double, double> result;
	for(int i = 0; i < baseline.size(); i++){
		if(metric == "abs"){
			result.first += abs( baseline[i][1] - actual[i][1] );
			result.second += abs( baseline[i][2] - actual[i][2] );
		}
		else{
			result.first += ( baseline[i][1]-actual[i][1] ) * ( baseline[i][1] - actual[i][1] );
			result.second += ( baseline[i][2]-actual[i][2] ) * ( baseline[i][2] - actual[i][2] );
		}
	}
	result.first /= baseline.size();
	result.second /= baseline.size();
	if(metric == "rms"){
		result.first = sqrt(result.first);
		result.second = sqrt(result.second);
	}
	// Return a pair of accuracies of Queue and Dynamic Densities
	return {(double)round(10000*(1-result.first))/100, (double)round(10000*(1-result.second))/100};
}

// Loads a result file (with frame data) into a vector for processing
vector<vector<double>> load_file(string file_name = "baseline.txt"){

	fstream f("../Analysis/Outputs/"+file_name);
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