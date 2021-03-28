#include "helpers.hpp"

// Function to get baseline result
void baseline(string file_name);

int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "Incorrect number of arguments (refer to README.md)." << endl;
        return 0;
    }
    baseline(argv[1]);

	return 0;
}

// Function to get baseline result
void baseline(string file_name){

    VideoCapture cap;

    cap.open("Videos/"+file_name); // Open video file
    Mat empty = imread("Images/empty.jpg"); // Open background image

    auto start = high_resolution_clock::now(); // Start clock to get run-time

    // Extract frame data
    vector<vector<double>> result = getDensityData(cap);

    // vector<vector<double>> result = getDensityDataTemporal(cap, empty);
     
    // --------------remember changing i = 0 / 1 loop at line 43

    cap.release(); // Close the VideoCapture

    auto stop = high_resolution_clock :: now(); // Stop clock
    auto duration = duration_cast<microseconds> (stop - start); // Get duration


    // Output result to a file
    string output_file = "Outputs/baseline_test.txt";
    fstream f(output_file, ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

    for(int i = 1 ; i < result.size() ; i++ ) {
        cout << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
        f << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
    }

    // Append the time taken for analysis
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    cout << "Output saved to " << output_file << endl;
    f.close();
}
