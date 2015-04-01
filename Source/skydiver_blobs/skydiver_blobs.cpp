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

void rotate_mat(Mat& src, Mat& dst, double angle, double scale=1)
{
    Point2f center = Point2f(src.rows/2, src.cols/2);
    Mat rotMatrix = getRotationMatrix2D(center, angle, scale);
    warpAffine(src, dst, rotMatrix, src.size());
}

vector<Point2f> rotate_pts(vector<Point2f>& pts, double angle, double scale=1, bool use_radians=false)
{
    if(!use_radians)
        angle *= (3.141592654/180);

    vector<Point2f> dst;
    Point2f mean = get_vec_centroid(pts);

    for(int i=0; i<pts.size(); i++)
    {
        Point2f pt = pts[i] - mean;
        Point2f tmp;
        cout << "pt" << i << ": " << pt.x << ", " << pt.y << endl;
        pt *= scale;
        tmp.x = pt.x*cos(angle) - pt.y*sin(angle);
        tmp.y = pt.x*sin(angle) + pt.y*cos(angle);
        tmp += mean;
        dst.push_back(tmp);
    }

    return dst;
}

int main()
{
    Mat dst, src = imread("skydiver_blobs_unconnected1.png", CV_LOAD_IMAGE_GRAYSCALE);
    vector<vector<Point> > skydiverBlobContours;

    PCA pca;
    PCA_load(pca, "../../Data/out/PCA.yml");

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

    Mat meanPointsMat(meanPoints);

    vector<vector<Point2f> > GPAPoints;
    if(!load_data_pts("data_points_mean.txt", GPAPoints))
        cout << "Could not open mean data points file." << endl;

    vector<Mat> GPAPointsMat;
    for(int i=0; i< GPAPoints.size(); i++)
        GPAPointsMat.push_back(Mat(GPAPoints));

//    Mat GPAPointsMatPCA = formatImagesForPCA(GPAPointsMat);
//    Mat projected =  pca.project(GPAPointsMatPCA.row(0));



    double meanScaleMetric = get_scale_metric(meanPointsMat);
    Point2f meanCentroid = get_vec_centroid(meanPoints);
    double meanOrientation = get_major_axis(meanPointsMat);

    cout << "meanScaleMetric: " << meanScaleMetric;
    cout << ", meanCentroid: " <<  meanCentroid;
    cout << ", meanOrientation: " << meanOrientation << endl;

    vector<vector<Point2f> > initialModelFit(4);

    for(int i=0; i<4; i++) //For each skydiver blob
    {
        double initialScale = skydivers[i].scaleMetric/meanScaleMetric;
        Point2f initialTranslation = skydivers[i].centroid - meanCentroid;
        double initialRotation = skydivers[i].orientation - meanOrientation;

        Mat initialModelFitMat = meanPointsMat.clone();
        initialModelFitMat += Scalar(initialTranslation.x, initialTranslation.y);

        initialModelFitMat.reshape(2).copyTo(initialModelFit[i]);

        initialModelFit[i] = rotate_pts(initialModelFit[i], initialRotation, initialScale);

        //Output
        Mat params = skydivers[i].paramaters_image().clone();
        Scalar colour(128);
        draw_body_pts(params, initialModelFit[i], colour);

        imshow("Window", params);

        stringstream numSS("");
        numSS << i;
        imwrite("blob_params_" + numSS.str() + ".jpg", params);
        waitKey(0);
    }

   return 0;
}








