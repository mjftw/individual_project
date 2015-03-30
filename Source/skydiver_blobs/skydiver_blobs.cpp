#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "../mw_libCV.h"
#include "skydiverblob.h"

#define SRC_VID_DIR (std::string)"../../Data/src/vid/"
#define SRC_IMG_DIR (std::string)"../../Data/src/img/"
#define OUT_DIR (std::string)"../../Data/out/"

#define FCC CV_FOURCC('P','I','M','1') //MPEG-1 codec compression
//#define FCC 0 //uncompressed

#define BLACK 0
#define WHITE 255

using namespace std;
using namespace cv;

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

int main()
{
    Mat dst, src = imread("skydiver_blobs_unconnected1.png", CV_LOAD_IMAGE_GRAYSCALE);
    vector<vector<Point> > skydiverBlobContours;

    namedWindow("Window", CV_WINDOW_NORMAL);
    imshow("Window", src);
    waitKey(0);

    find_skydiver_blobs(src, skydiverBlobContours);

    overlay_contours(src, dst, skydiverBlobContours);
    imshow("Window", dst);
    imwrite("skydiver_contours.jpg", dst);
    waitKey(0);

    Mat skydiverBlobs[4];
    SkydiverBlob skydivers[4];

    for(int i=0; i<4; i++)
        skydivers[i].approx_parameters(skydiverBlobContours[i], src.rows, src.cols);

    vector<vector<Point2f> > meanPointsVec;
    if(!load_data_pts("data_points_mean.txt", meanPointsVec))
        cout << "Could not open mean data points file." << endl;
    vector<Point2f> meanPoints = meanPointsVec[0];

    double meanScaleMetric = get_scale_metric(meanPoints);
    Point2f meanCentroid = get_vec_centroid(meanPoints);
    Mat meanPointsMat(meanPoints);
    double meanOrientation = get_major_axis(meanPointsMat);

    cout << "meanScaleMetric: " << meanScaleMetric;
    cout << ", meanCentroid: " <<  meanCentroid;
    cout << ", meanOrientation: " << meanOrientation << endl;

    double initialScale[4];
    Point2f initialTranslation[4];
    double initialRotation[4];
    Mat initialModelFitMat[4];
    vector<vector<Point2f> > initialModelFit(4);


    for(int i=0; i<4; i++) //For each skydiver blob
    {
        initialScale[i] = skydivers[i].scaleMetric/meanScaleMetric;
        initialTranslation[i] = skydivers[i].centroid - meanCentroid;
        initialRotation[i] = skydivers[i].orientation - meanOrientation;

        initialModelFitMat[i] = meanPointsMat;
//        initialModelFitMat[i] *= initialScale[i];
        initialModelFitMat[i] += Scalar(initialTranslation[i].x, initialTranslation[i].y);

        cout << "scale: " << initialScale[i];
        cout << ", translation: " << initialTranslation[i];
        cout << ", rotation: " << initialRotation[i] << endl;

        cout << initialModelFitMat[i] << endl;

        initialModelFitMat[i].reshape(2).copyTo(initialModelFit[i]);

        //Output
        Mat params = skydivers[i].paramaters_image();
        Scalar colour(128);
        plot_pts(params, initialModelFit[i], colour);

        imshow("Window", params);

        stringstream numSS("");
        numSS << i;
        imwrite("blob_params_" + numSS.str() + ".jpg", params);
        waitKey(0);
    }

   return 0;
}








