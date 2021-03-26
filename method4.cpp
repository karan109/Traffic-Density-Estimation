#include "helpers.hpp"

#define THREADS 6

int img_size;
struct thread_data {
    VideoCapture cap;
    Mat empty;
    int part;
    int th;
};
VideoCapture cap;
thread threads[THREADS];
vector<vector<vector<double>>> result(THREADS, vector<vector<double>>(0, vector<double>(0)));
vector<vector<double>> final_result;
void temporal(void* arg) {
    struct thread_data *my_data;
	my_data = (struct thread_data *) arg;
	Mat frame_empty = my_data->empty;
	int part = my_data->part;
	int num = THREADS;
	int step = 3;
    result[part-1] = getDensityDataTemporal(my_data->cap, my_data->empty, step, my_data->part, THREADS, my_data->th);
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
		data[i].th = i+1;
        high_resolution_clock::time_point start = high_resolution_clock::now();
        threads[i] = thread(temporal, (void*)&data[i]);
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
    for(auto part:result){
        for(auto row:part){
            final_result.push_back(row);
        }
    }
    

    cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;
    fstream f("Outputs/Method4/"+to_string(THREADS)+".txt", ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;


    for(int i=0;i<final_result.size();i++){
        cout << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
        f << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
    }
    f << duration.count();
    f.close();

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    vector<vector<double>> baseline = load_file();
    pair<double, double> accuracy = getAccuracy(final_result, baseline, "abs");
    cout << accuracy.first << " " << accuracy.second << " " << endl;
}