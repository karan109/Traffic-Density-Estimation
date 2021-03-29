#include "helpers.hpp"

int img_size; // Size of cropped image

// Struct definition of arguments passed to each thread
struct thread_data {
   VideoCapture cap;
   Mat empty;
   int part;
};

int THREADS; // Number of threads

vector<vector<vector<double>>> result; // Stores results of each thread as a vector
vector<vector<double>> final_result; // Combines the individual results into final result

// Function declaration of method 3
void method3(string file_name);

// Function to pass to each thread
void * spatial(void * arg) {
    // Argument to thread
    struct thread_data * my_data;
	my_data = (struct thread_data * ) arg;

	Mat frame_empty = my_data->empty;
	int part = my_data->part;
	int step = 1;

    // Get individual result of thread
    auto temp = getDensityDataSpatial(my_data->cap, my_data->empty, step, my_data->part, THREADS);
	result[part-1] = temp.first;
    img_size = temp.second;

    // Return a value to circumvent warnings
    return (void *) 0;
}

int main(int argc, char* argv[]){
    if(argc != 3){
        cout << "Incorrect number of arguments (refer to README.md)." << endl;
        return 0;
    }
    if(!isint(argv[2])){
        cout << "Number of threads given is not an integer (refer to README.md)." << endl;
        return 0;
    }
    THREADS = stoi(argv[2]);
    method3(argv[1]);

    return 0;
}

// Function declaration of method 3
void method3(string file_name){

    // Array of pthreads
    pthread_t threads[THREADS];

    result.assign(THREADS, vector<vector<double>>(0, vector<double>(0)));

    // Array of thread arguments
    struct thread_data data[THREADS];

    auto start = high_resolution_clock::now(); // Start clock to calculate time

    // Evaluate each thread
    for (int i=0;i<THREADS;i++) {
        data[i].cap.open("Videos/"+file_name);

        if (data[i].cap.isOpened() == false){
            cout << "Cannot open the video file. Please provide a valid name (refer to README.md)." << endl;
            exit(3);
        }

        data[i].empty = imread("Images/empty.jpg");
        data[i].part = i+1;
        pthread_create(& threads[i], NULL, spatial, (void * ) & data[i]);
    }

    for(int i=0;i<THREADS;i++){
        pthread_join(threads[i], NULL); // Wait for each thread to finish
    }

    for(int i=0;i<THREADS;i++){
        data[i].cap.release(); // Close VideoCapture of each thread
    }

    auto stop = high_resolution_clock :: now(); // Stop clock
    auto duration = duration_cast<microseconds> (stop - start); // Get duration


    // Compute the final result by taking the weighted average of all densities returned by individual threads
    final_result.assign(result[0].size(), vector<double>(result[0][0].size(), 0));
    for(int i = 0; i < result.size(); i++){
        for(int j = 0; j < result[i].size(); j++){
            for(int k = 1; k < result[i][j].size(); k++){
                final_result[j][k] += result[i][j][k];
            }
            final_result[j][0] = result[0][j][0];
        }
    }
    for(int j = 0; j < final_result.size(); j++){
        for(int k = 1; k < final_result[j].size(); k++){
            final_result[j][k] /= img_size;
        }
    }

    // Output result to a file
    string output_file = "Outputs/Method3/"+to_string(THREADS)+"_test.txt";
    fstream f(output_file, ios::out);
    f << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

    // cout << "Frame_Num,Queue_Density,Dynamic_Density" << endl;

    for(int i=0;i<final_result.size();i++){
        // cout << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
        f << final_result[i][0] << "," << final_result[i][1] << "," << final_result[i][2] << endl;
    }

    // Append the time taken for analysis
    f << duration.count();

    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
    cout << "Output saved to " << output_file << endl;

    f.close();
}