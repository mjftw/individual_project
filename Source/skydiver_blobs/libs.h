#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

#define WHITE 255
#define BLACK 0

inline bool contourSort(const vector<Point> contour1, const vector<Point> contour2)
{
    return (contourArea(contour1, false) > contourArea(contour2, false));
}

void find_skydiver_blobs(Mat& binary_src, vector<vector<Point> >& skydiver_blob_contours)
{
    vector<vector<Point> > connectedComponents;
    Mat temp;

    skydiver_blob_contours.clear();
    threshold(binary_src, temp, 100, WHITE, THRESH_BINARY);

    findContours(temp, connectedComponents, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
    sort(connectedComponents.begin(), connectedComponents.end(), contourSort);

    //*TODO* Need to make this work if less than 4 blobs.
    //This will be the case if the skydivers are too close & their blobs overlap
    for(int i=0;i<4; i++)
        skydiver_blob_contours.push_back(connectedComponents[i]);

    binary_src = Scalar(BLACK);
    drawContours(binary_src, skydiver_blob_contours, -1, Scalar(WHITE), -1);

    return;
}
