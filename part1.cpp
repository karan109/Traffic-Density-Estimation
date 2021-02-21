// Includes
#include "opencv2/opencv.hpp" 
#include <vector>

using namespace cv;
using namespace std;

// Global Variables

int num_points = 0; // Number of points selected by user
vector<Point2f> points(4); // Vector to store 4 selected points


// Constants

// Points given by ma'am in the specifications
const vector<Point2f> GIVEN_POINTS = {Point2f(472,52), Point2f(472,830), Point2f(800,830), Point2f(800,52)};

// (x1, y1, width, height) of the rectangle formed by ma'am's points 
const Rect RECT_CROP(GIVEN_POINTS[0].x, GIVEN_POINTS[0].y, GIVEN_POINTS[2].x - GIVEN_POINTS[0].x, GIVEN_POINTS[1].y - GIVEN_POINTS[0].y);

// Functions

// Print current selection of points
void printSelection();

// Callback function to detect left mouse clicks
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Transform and Crop original image after converting to grayscale
int transformCrop(string imagePath);

// Main
int main(int argc, char* argv[]){
    String image = (argc == 1) ? "empty.jpg" : argv[1];
    transformCrop(image);
    return 0;
}


void printSelection(){
    cout << '[';
    if(num_points > 0) cout << '(' << points[0].x << ", " << points[0].y << ')';
    for(int i = 1; i < num_points; i++){
        cout << ", (" << points[i].x << ", " << points[i].y << ")";
    }
    cout << ']' << endl;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata){
     if(event == EVENT_LBUTTONDOWN){
        points[num_points++] = Point2f(x, y);

        // Print current selection of points (for debugging)
        cout << "Mouse click detected. Current selection: ";
        printSelection();
    }
}

int transformCrop(string imageName){

    // Declarations for images
    Mat im_original, im_gray, homography, im_transform, im_crop;

    // Read from the command-line argument
    im_original = imread("Images/"+imageName);

    // Convert to grayscale and store in im_gray
    cvtColor(im_original, im_gray, COLOR_BGR2GRAY);

    // Make a window for im_gray
    String window_name = "Grayscale";
    namedWindow(window_name, WINDOW_NORMAL);

    // Show grayscale image
    imshow(window_name, im_gray);

    // Prompt to select 4 points
    cout << "Please select 4 points in a counter-clockwise manner starting from the top left." << endl;

    // Wait for user mouse clicks
    while(true){
        if(num_points < 4) setMouseCallback(window_name, CallBackFunc, NULL);
        else setMouseCallback(window_name, NULL, NULL);
        
        char c = waitKey(10);
        if(c == 13){
            // Enter key detected
            if(num_points == 4){
                cout << "Final selection: ";
                printSelection();
                break;
            }
            else{
                cout << "Please select remaining points. Current selection: ";
                printSelection();
            }
        }
        if(c == 'u'){
            // User wants to undo the point clicked if 'u' is pressed
            num_points--;
            cout << "UNDO done. Current selection: ";
            printSelection();
        }
        if(c == 'r'){
            // User wants to reset all points and start from beginning if 'r' is pressed
            num_points = 0;
            cout << "RESET done. Current selection: ";
            printSelection();
        }
        if(c == 27){
            // Escape key pressed
            cout << "\"Esc\" key pressed. Aborting..." << endl;
            return -1;
        }
    }

    // 4 points are selected at this stage

    // Result of Homography stored in im_transform
    homography = findHomography(points, GIVEN_POINTS);
    warpPerspective(im_gray, im_transform, homography, im_gray.size());
    imshow("Transformed", im_transform);

    // Crop im_transform to get im_crop
    im_crop = im_transform(RECT_CROP);
    imshow("Cropped", im_crop);

    // Press Escape or Enter key to exit
    while(true){
        char c = waitKey(10);
        if(c == 13){
            // Save Images
            bool check_transform = imwrite("Transforms/transform_" + imageName, im_transform);
            bool check_crop = imwrite("Crops/crop_" + imageName, im_crop);

            // Succesfully saved
            if(check_crop == true and check_transform == true)
                cout << "Images saved in folders \"Crops\" and \"Transforms\"" << endl;
            
            // Unsuccessful
            else{
                cout << "Failed to save image(s). Please try again" << endl;
                return -1;
            }
            break;
        }
        if(c == 27){
            cout << "\"Esc\" key pressed. Aborting..." << endl;
            return -1;
        }
    }
    return 0;
}
