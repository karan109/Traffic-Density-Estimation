#include "opencv2/opencv.hpp"
#include <vector>
#include <chrono>
#include <thread>
#include "helpers.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;
#define THREADS 6

int img_size;
struct thread_data {
   VideoCapture cap;
   Mat empty;
   int part;
   string output;
   int th;
};
VideoCapture cap;
thread threads[THREADS];
vector<vector<vector<double>>> result(THREADS, vector<vector<double>>(0, vector<double>(0)));
vector<vector<double>> final_result;
void blurSlowdown(void* arg) {
    struct thread_data *my_data;
	my_data = (struct thread_data *) arg;
	Mat frame_empty = my_data->empty;
	string output = my_data->output;
	int part = my_data->part;
	int num = THREADS;
	int step = 3;
    auto temp = getDensityDataCustom(my_data->cap, my_data->empty, 3, my_data->part, THREADS, my_data->th, 0);
	result[part-1] = temp.first;
    img_size = temp.second;
}

int main()
{
	struct thread_data data[THREADS];
	auto start = high_resolution_clock::now();
    for (int i=0;i<THREADS;i++) {
    	string file_name = "trafficvideo.mp4";
    	data[i].cap.open("Videos/"+file_name);
    	data[i].empty = imread("Images/empty.jpg");
		data[i].part = i+1;
		data[i].output = "out"+to_string(i+1)+".txt";
		data[i].th = i+1;
        high_resolution_clock::time_point start = high_resolution_clock::now();
        threads[i] = thread(blurSlowdown, (void*)&data[i]);
        high_resolution_clock::time_point end = high_resolution_clock::now();
    }
    for(int i=0;i<THREADS;i++){
    	threads[i].join();
    }
    for(int i=0;i<THREADS;i++){
    	data[i].cap.release();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    final_result.assign(result[0].size(), vector<double>(result[0][0].size(), 0));
    for(int i=0;i<result.size();i++){
        for(int j=0;j<result[i].size();j++){
            for(int k=1;k<result[i][j].size();k++){
                final_result[j][k] += result[i][j][k];
            }
            final_result[j][0] = result[0][j][0];
        }
    }
    for(int j=0;j<final_result.size();j++){
        for(int k=1;k<final_result[j].size();k++){
            final_result[j][k] /= img_size;
        }
    }
    fstream f("Outputs/Method3/"+to_string(THREADS)+".txt", ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    for(int i=0;i<final_result.size();i++){
        cout << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
        f << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
    }
    f << duration.count();
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    vector<vector<double>> baseline = load_file();
    pair<double, double> accuracy = getAccuracy(final_result, baseline, "abs");
    cout << accuracy.first << " " << accuracy.second << " " << endl;
    f.close();
}