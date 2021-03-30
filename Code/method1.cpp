#include "helpers.hpp"

int X; //number of frames to skip
string file_name = "trafficvideo.mp4";
vector<vector<double>> result;
string output_file;

// Function declaration of method 1
void method1();

int main(int argc, char* argv[]){
    if(argc != 3){
        cout << "Incorrect number of arguments (refer to README.md)." << endl;
        return 0;
    }
    if(!isint(argv[2]) || stoi(argv[2]) <= 0 ){
        cout << "Number of frames to skip given is not a positive integer." << endl;
        return 0;
    }
    X = stoi(argv[2]);
    file_name = argv[1];
    method1();
}

void method1 () {

	VideoCapture cap;
	cap.open("../Data/Videos/"+file_name); // Capture video

	// if not success, exit program
	if (cap.isOpened() == false) {
		cout << "Cannot open the video file. Please provide a valid name (refer to README.md)." << endl;
        exit(3);
	}

	auto start = high_resolution_clock::now(); // Start clock to get run-time
    
    // Extract frame data
    result = getDensityDataSkips(cap, X);

    cap.release(); // Close the VideoCapture

    auto stop = high_resolution_clock :: now(); // Stop clock
    auto duration = duration_cast<microseconds> (stop - start); // Get duration


	// Output file
	output_file = "../Analysis/Outputs/Method1/"+to_string(X)+"_test.txt";

    fstream f(output_file, ios::out);
    
	f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

	for(int i = 1 ; i < result.size() ; i++) { 
        f << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
    }

    // Append the time taken for analysis
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    cout << "Output saved to " << output_file << endl;
    f.close();
}







