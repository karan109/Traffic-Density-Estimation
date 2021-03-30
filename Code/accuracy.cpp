#include "helpers.hpp"

int main(){
    vector<vector<double>> baseline = load_file("baseline.txt");
    vector<vector<double>> actual = load_file("Method2/1600x900.txt");
    pair<double, double> accuracy = getAccuracy(actual, baseline, "abs");
    cout << accuracy.first << " " << accuracy.second << " " << endl;
	return 0;
}