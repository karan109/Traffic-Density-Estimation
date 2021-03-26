// Includes
#include "helpers.hpp"

int getDensityData(string file_name);

int main(int argc, char* argv[]){
	// Command line argument
	string video = (argc == 1) ? "trafficvideo.mp4" : argv[1];
	getDensityData(video);
	return 0;
}

int getDensityData(string file_name){
	// Find homography using a default set of points
	default_homography = findHomography(DEFAULT_POINTS, GIVEN_POINTS);

	VideoCapture cap("Videos/"+file_name); // Capture video

	int fps = cap.get(CAP_PROP_FPS); // Get fps

	// if not success, exit program
	if (cap.isOpened() == false){
		cout << "Cannot open the video file. Please provide a valid name (refer to README.md)." << endl;
		return -1;
	}

	// Output file
	string output_name = "Outputs/user_out.txt";
	fstream f(output_name, ios::out);

	cout << "Processing... (Program output will be saved to \"" + output_name + "\")" << endl;

	int frame_count = 0;
	double total_density = 0, dynamic_density = 0; // Average densities over 15 (fps) frames per second
	Mat frame_old_difference; // Old frame too get optical flow

	cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
	f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

	while(true){

		Mat frame_current, frame_processed, frame_empty, frame_difference, frame_threshold;
		bool success = cap.read(frame_current); // read a new frame from video

		// Exit if no more frames available
		if(success == false){
			cout << "Found the end of the video. Density data saved at \"" + output_name + "\"" << endl;
			break;
		}
		frame_processed = pre_process(frame_current, default_homography); // Apply transformation and crop
		
		frame_empty = imread("Images/empty.jpg"); // Get background image
		frame_empty = pre_process(frame_empty, default_homography); // Apply transformation and crop to background


		absdiff(frame_processed, frame_empty, frame_difference); // Background subtraction
		threshold(frame_difference, frame_threshold, 40, 255.0, THRESH_BINARY);
		filter(frame_threshold); // Smoothen and fill gaps

		// vector<vector<Point> > contours;
		// vector<Vec4i> hierarchy;
		// findContours(imgThresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Set the old frame to the current frame in case of frame 0
		if(frame_count == 0) frame_old_difference = frame_difference;

		// flow is an image where all moving pixels are white and all stationary pixels are black
		Mat flow = getFlow(frame_old_difference, frame_difference);
		threshold(flow, flow, 23, 255.0, THRESH_BINARY);
		filter(flow); // Smoothen and fill gaps

		int size = frame_processed.size().height * frame_processed.size().width; // Size of frame
		double pixel_ratio = (double) countNonZero(frame_threshold) / size; // To get density
		double dynamic_pixel_ratio = (double) countNonZero(flow) / size; // To get dynamic density

		// Show the densities
		imshow("Dynamic density", flow);
		imshow("Queue Density", frame_threshold);

		total_density += pixel_ratio;
		dynamic_density += min(dynamic_pixel_ratio, pixel_ratio);

		// Output data for every 3 frames
		int step = 3;

		if(frame_count % step == 0 and frame_count != 0){

			// Every step, evaluate average densities
			dynamic_density /= step;
			total_density /= step;

			cout << frame_count << "," << total_density << "," << min(dynamic_density, 0.95 * total_density) << endl;
			f << frame_count << "," << total_density << "," << min(dynamic_density, 0.95 * total_density) << endl;
			
			// Reset current densities to 0 to prepare for the next second
			total_density = 0;
			dynamic_density = 0;
		}

		char c = waitKey(10);
		if(c == 27){
			// Escape key detected
			cout << "\"Esc\" key detected. Partial density data saved at \"" + output_name + "\"" << endl;
			f.close();
			break;
		}

		frame_count++; // Update frame count

		frame_old_difference = frame_difference; // Update old frame
	}
	f.close();
	cout << "Video processed. Density data saved at \"" + output_name + "\"" << endl;
	return 0;
}