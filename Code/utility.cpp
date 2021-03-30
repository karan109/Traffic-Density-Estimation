#include "helpers.hpp"

int main(int argc, char* argv[]){
	if(argc != 2){
        cout << "Incorrect number of arguments (refer to README.md)." << endl;
        return 0;
    }
    string input_file = argv[1];

    vector<vector<double>> baseline = load_file("baseline.txt");
    vector<vector<double>> actual = load_file(input_file);
    pair<double, double> accuracy = getAccuracy(actual, baseline, "abs");
    cout << "Queue: " << accuracy.first << ", Dynamic: " << accuracy.second << " " << endl;
	return 0;
}