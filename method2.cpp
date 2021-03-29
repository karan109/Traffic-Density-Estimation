#include "helpers.hpp"

string file_name = "trafficvideo.mp4";
int X;
int Y;
int original_X;
int original_Y;
string output_file;
fstream f(output_file, ios::out);
vector<vector<double>> result;


void method2();

int main(int argc, char* argv[]){

    if (argc >= 2) file_name = argv[1];

    VideoCapture getres;
    getres.open("Videos/"+file_name);
    Mat temp;
    getres.read(temp);
    original_Y = temp.rows;
    original_X = temp.cols;
    X = original_X;
    Y = original_Y;

    getres.release(); 

    if (argc >= 3) {
        if(!isint(argv[2]) || stoi(argv[2]) <= 0 ) {
            cout << "X resolution is not a positive integer." << endl;
            return 0;
        }
        X = stoi(argv[2]);        
    }
    if (argc == 4) {
        if(!isint(argv[3]) || stoi(argv[3]) <= 0 ) {
            cout << "Y resolution is not a positive integer." << endl;
            return 0;
        }
        Y = stoi(argv[3]);     
    }

    method2();
}

void method2 () {

    VideoCapture cap;
    cap.open("Videos/"+file_name); // Capture video

    // if not success, exit program
    if (cap.isOpened() == false) {
        cout << "Cannot open the video file. Please provide a valid name (refer to README.md)." << endl;
        exit(1);
    }

    auto start = high_resolution_clock::now(); // Start clock to get run-time
    
    // Extract frame data

    // result = getDensityDataResolutionEasy(cap, X, Y);
    result = getDensityDataResolution(cap, X, Y, original_X, original_Y);

    cap.release(); // Close the VideoCapture

    auto stop = high_resolution_clock :: now(); // Stop clock
    auto duration = duration_cast<microseconds> (stop - start); // Get duration


    // Output file
    output_file = "Outputs/Method2/["+to_string(X)+ " x " + to_string(Y) + "].txt";

    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

    for(int i = 1 ; i < result.size() ; i++) { 
        cout << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
        f << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
    }

    // Append the time taken for analysis
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    cout << "Output saved to " << output_file << endl;
    f.close();
}

