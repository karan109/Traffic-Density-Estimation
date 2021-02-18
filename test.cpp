#include "opencv2/opencv.hpp" 
#include <vector>
using namespace cv;
using namespace std;

int ct = 0; // Number of points selected by user
vector<pair<int, int> > points(4); // 4 points selected by user

// To detect mouse clicks by user
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
        points[ct++] = make_pair(x, y);
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        
        // Print current selection of points (for debugging)
        for(int i=0;i<ct;i++){
            cout<<points[i].first<<" "<<points[i].second<<endl;
        }

        // Print number selected of points
        cout<<ct<<endl;
     }
}

int main(int argc, char* argv[])
{
    
    Mat im_gray, im_src = imread("traffic.jpg");

    // Convert to grayscale
    cvtColor(im_src, im_gray, COLOR_BGR2GRAY);

    String window_name = "Original";
    namedWindow(window_name, WINDOW_NORMAL);
    setMouseCallback(window_name, CallBackFunc, NULL);

    // Show grayscale image
    imshow(window_name, im_gray);

    // Wait for user mouse clicks
    while(true){
        if(ct == 4){
            // 4 points selected, now break
            setMouseCallback(window_name, NULL, NULL);
            break;
        }

        // 4 points are selected by user starting from the top left in anti-clockwise manner
        char c = waitKey(10);
        if(c == 'u') ct--; // User wants to undo the point clicked if 'u' is pressed
        if(c == 'r') ct = 0; // User wants to reset all points and start from beginning if 'r' is pressed
    }
    vector<Point2f> pts_src;
    for(int i=0;i<4;i++){
        pts_src.push_back(Point2f(points[i].first, points[i].second));
    }

    vector<Point2f> pts_dst;

    // 4 points given by maam
    pts_dst.push_back(Point2f(472,52));
    pts_dst.push_back(Point2f(472,830));
    pts_dst.push_back(Point2f(800,830));
    pts_dst.push_back(Point2f(800,52));

    // Result of Homography stored in im_temp
    Mat im_temp;
    Mat h = findHomography(pts_src, pts_dst);
    warpPerspective(im_gray, im_temp, h, im_gray.size());
    imshow("Projection", im_temp);

    cout<<im_gray.size()<<endl;

    // Crop im_temp to get crop
    Rect roi(472, 52, 328, 778); // (x1, y1, width, height) of the rectangle formed by ma'am's points 
    Mat crop = im_temp(roi);
    imshow("Cropped Image", crop);

    // Press escape key to exit
    while(true){
        char c = waitKey(10);
        if(c == 27) break;
    }
    return 0;
}