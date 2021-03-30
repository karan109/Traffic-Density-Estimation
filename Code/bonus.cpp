#include "helpers.hpp"

// Function to get baseline result
void bonus(string file_name);

int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "Incorrect number of arguments (refer to README.md)." << endl;
        return 0;
    }
    bonus(argv[1]);

	return 0;
}

// Function to get baseline result
void bonus(string file_name){

    Mat empty = imread("../Data/Images/empty.jpg"); // Open background image

    auto start = high_resolution_clock::now(); // Start clock to get run-time

    // Extract frame data
    vector<vector<double>> result = getDensityDataSparse(file_name, empty);


    auto stop = high_resolution_clock :: now(); // Stop clock
    auto duration = duration_cast<microseconds> (stop - start); // Get duration


    // Output result to a file
    string output_file = "../Analysis/Outputs/bonus_test.txt";
    fstream f(output_file, ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

    for(int i=0;i<result.size();i++){
        cout << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
        f << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
    }

    // Append the time taken for analysis
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    cout << "Output saved to " << output_file << endl;
    f.close();
}