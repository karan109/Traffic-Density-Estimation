// Includes
#include "opencv2/opencv.hpp"
#include <vector>
#include <fstream>
#include <math.h>
#include <chrono>

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

// Transform and return cropeed image without any manual input from user
Mat pre_process(Mat img, Mat homography, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop, im_gray = grayScale(img);
	warpPerspective(im_gray, im_transform, homography, im_gray.size());
	im_crop = im_transform(RECT_CROP);
	return im_crop;
}

Mat pre_process_(Mat img, Mat homography, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop;
	warpPerspective(img, im_transform, homography, img.size());
	im_crop = im_transform(RECT_CROP);
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

Mat getSparseFlow(Mat & frame_old_difference, Mat & frame_difference, Mat & temp){
	vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    vector<Point2f> p0, p1;
    goodFeaturesToTrack(frame_difference, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
	Mat mask = Mat::zeros(temp.size(), temp.type());
	vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(frame_old_difference, frame_difference, p0, p1, status, err, Size(15,15), 2, criteria);
    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask,p1[i], p0[i], colors[i], 2);
            circle(temp, p1[i], 5, colors[i], -1);
        }
    }
    Mat img;
    add(temp, mask, img);
    img = grayScale(img);
    return img;
	// return getFlow(frame_old_difference, frame_difference);
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

	Mat frame0;

	// Begin processing from actual_start
	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);
		if(success == false) break;
		frame_processed = pre_process(frame_current, default_homography);

		absdiff(frame_processed, frame_empty, frame_difference);
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold);

		if(frame_count == 0){ frame_old_difference = frame_difference; frame0 = frame_current;}

		Mat temp = pre_process_(frame0, default_homography);

		Mat flow = getSparseFlow(frame_old_difference, frame_difference, temp);
		threshold(flow, flow, 230, 255.0, THRESH_BINARY);
		// filter(flow);

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
		// imshow("dynamic", flow);
		// Skip every step number of frames
		if((frame_count) % step == 0 and frame_count != 0){
			dynamic_density /= step;
			total_density /= step;

			// Add frame data to result vector
			result.push_back( { (double)frame_count, total_density, min(dynamic_density, 0.95 * total_density) } );

			cout << frame_count << "," << total_density << "," << min(dynamic_density, 0.95 * total_density) << endl;

			total_density = 0;
			dynamic_density = 0;
		}

		frame_count++;

		frame_old_difference = frame_difference;
	}
	cap.release(); // Close the VideoCapture
	return result;
}



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
	return {(double)round(10000*(1-tanh(result.first)))/100, (double)round(10000*(1-tanh(result.second)))/100};
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

// Function to check if given string is an integer
int isint(string var)
{

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

vector<vector<double>> getDensityDataSkips (VideoCapture &cap, int X = 1) {

	vector<vector<double>> result;

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS); 
	Mat frame_empty = imread("../Data/Images/empty.jpg");
	frame_empty = pre_process(frame_empty, default_homography);
	Mat frame_prev;
	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	while(true) {

		Mat frame_current; 
		Mat frame_processed;
		Mat frame_prev_processed; 
		Mat frame_threshold;
		Mat frame_difference; // need to be local or clear it 
		Mat frame_prev_difference; // need to be local or clear it 

		bool success = cap.read(frame_current);
		if (success == false) break;

		if (frame_count % X != 0) {
			result.push_back( { (double) frame_count, result[frame_count - 1][1], result[frame_count - 1][2] });

			// print_result(result, frame_count);	
			
			frame_count++;
			frame_prev = frame_current;
			continue;
		}

		frame_processed = pre_process(frame_current, default_homography); 
		absdiff(frame_processed, frame_empty, frame_difference); 

		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); 

		if(frame_count == 0) {
			frame_prev = frame_current;
		}

		frame_prev_processed = pre_process(frame_prev, default_homography); 
		absdiff(frame_prev_processed, frame_empty, frame_prev_difference); 

		Mat flow = getFlow(frame_prev_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow);

		int size = frame_processed.size().height * frame_processed.size().width; // Size of frame
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

vector<vector<double>> getDensityData(VideoCapture &cap) {

	vector<vector<double>> result;

	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	Mat frame_old_difference;
	Mat frame_empty = imread("../Data/Images/empty.jpg");
	frame_empty = pre_process(frame_empty, default_homography);

	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);

		if(success == false){
			break;
		}

		frame_processed = pre_process(frame_current, default_homography);

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

		// print_result(result, frame_count);	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}

vector<vector<double>> getDensityDataResolutionEasy(VideoCapture &cap, int X, int Y) {

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
		
		resize(frame_processed, frame_processed, dim);

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

		// print_result(result, frame_count);	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}






// ---------------------------------------------------------------------------------------------------------------------


vector<Point2f> GIVEN_POINTS_RES;
Rect RECT_CROP_RES;
vector<Point2f> DEFAULT_POINTS_RES;

vector<Point2f> scaling (const vector<Point2f> &a, double fx, double fy) {
	vector<Point2f> b;
	for (int i = 0; i < a.size() ; ++i) {
		b.push_back(Point2f(round (a[i].x * fx) , round (a[i].y * fy) ) );
	}
	return b;
}
Rect getRect (const vector<Point2f> &a) {
	Rect RECT_CROP(a[0].x, a[0].y, a[2].x - a[0].x, a[1].y - a[0].y);
	return RECT_CROP;
}

Mat pre_process(Mat img, Mat homography, Rect RECT_CROP, bool save=false, string imageName="empty.jpg"){
	Mat im_transform, im_crop, im_gray = grayScale(img);
	warpPerspective(im_gray, im_transform, homography, im_gray.size());
	im_crop = im_transform(RECT_CROP);
	return im_crop;
}

void print_points (vector<Point2f> &a) {
	for (int i = 0; i < a.size() ; ++i) {
		cout << "( " << a[i].x << ", " << a[i].y << ")" << "	";
	}
	cout << endl;
}


vector<vector<double>> getDensityDataResolution (VideoCapture &cap, int X, int Y, int original_X, int original_Y ) {

	Size dim(X, Y);

	double fx = (double)X / (double)original_X;
	double fy = (double)Y / (double)original_Y;

	// cout << original_X << " " << original_Y << " " << X << " " << Y << " " <<  fx << " " << fy << endl;

	GIVEN_POINTS_RES = scaling(GIVEN_POINTS, fx, fy);
	RECT_CROP_RES = getRect (GIVEN_POINTS_RES);
	DEFAULT_POINTS_RES = scaling(DEFAULT_POINTS, fx, fy);

	// print_points(GIVEN_POINTS_RES);
	// print_points(DEFAULT_POINTS_RES);


	vector<vector<double>> result;

	default_homography = findHomography(DEFAULT_POINTS_RES, GIVEN_POINTS_RES);

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0;

	Mat frame_old_difference;
	Mat frame_empty = imread("../Data/Images/empty.jpg");

	resize(frame_empty, frame_empty, dim);

	frame_empty = pre_process(frame_empty, default_homography, RECT_CROP_RES);


	while(true){

		Mat frame_current, frame_processed, frame_difference, frame_threshold;
		bool success = cap.read(frame_current);

		if(success == false){
			break;
		}
	
		resize(frame_current, frame_current, dim);

		frame_processed = pre_process(frame_current, default_homography, RECT_CROP_RES);

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

		// print_result(result, frame_count);	

		frame_count++;
		frame_old_difference = frame_difference;
	}
	return result;
}
