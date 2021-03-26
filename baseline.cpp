#include "helpers.hpp"

string file_name = "trafficvideo.mp4";
VideoCapture cap;
int main(){
	cap.open("Videos/"+file_name);
	Mat empty = imread("Images/empty.jpg");
	auto start = high_resolution_clock::now();
	auto temp = getDensityDataSpatial(cap, empty, 1, 1, 1);
	vector<vector<double>> result = temp.first;
    int img_size = temp.second;
	cap.release();
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    fstream f("Outputs/baseline.txt", ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    for(int j=0;j<result.size();j++){
        for(int k=1;k<result[j].size();k++){
            result[j][k] /= img_size;
        }
    }
    for(int i=0;i<result.size();i++){
        cout << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
        f << result[i][0] << "," << result[i][1] << "," << result[i][2] << endl;
    }
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    f.close();
	return 0;
}