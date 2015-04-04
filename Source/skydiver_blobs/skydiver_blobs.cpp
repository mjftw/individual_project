#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "../mw_libCV.h"
#include "../../external_libs/Procrustes/Procrustes.h"
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

bool find_4_skydiver_blobs(Mat& binary_src, Mat& dst, vector<vector<Point> >& skydiver_blob_contours)
{
    vector<vector<Point> > connectedComponents;
    Mat temp;

    skydiver_blob_contours.clear();
    threshold(binary_src, temp, 0, WHITE, THRESH_BINARY | THRESH_OTSU);

    findContours(temp, connectedComponents, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());

    if(connectedComponents.size() < 4)
        return false;

    sort(connectedComponents.begin(), connectedComponents.end(), contourSort);

    //*TODO* Need to make this work if less than 4 blobs.
    //This will be the case if the skydivers are too close & their blobs overlap
    for(int i=0;i<((connectedComponents.size() < 4)? connectedComponents.size():4) ; i++)
        skydiver_blob_contours.push_back(connectedComponents[i]);

    dst = binary_src.clone();
    dst = Scalar(BLACK);
    drawContours(dst, skydiver_blob_contours, -1, Scalar(WHITE), -1);

    if(connectedComponents.size() == 4)
        return true;
    else
        return false;
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
    VideoCapture srcVid(SRC_VID_PATH);
    if(!srcVid.isOpened())
    {
        cout << "Cannot open video file " << SRC_VID_PATH << endl;
        exit(EXIT_FAILURE);
    }

    PCA pca;
    PCA_load(pca, PCA_FILENAME);

    Mat frame, fgMask;
    Mat bg = imread(BG_IMG_PATH, CV_LOAD_IMAGE_GRAYSCALE);
    Mat skydiverBlobs;

    vector<vector<Point> > skydiverBlobContours;

//    Mat contourImg;
//    overlay_contours(skydiverBlobs, contourImg, skydiverBlobContours);
//    imshow("Window", skydiverBlobs);
//    waitKey(0);

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

    double meanScaleMetric = get_scale_metric(meanPointsMat);
    Point2f meanCentroid = get_vec_centroid(meanPoints);
    double meanOrientation = get_major_axis(meanPointsMat);


    namedWindow("PCA", WINDOW_NORMAL);
    Mat img(bg.size(), CV_8UC3, Scalar(0,0,0));



    vector<double> fakeP = {-3.815214, -0.671381, -0.262617, 0.594221, 0.0790387};
    vector<Point2f> dataOut;

    Mat fakePMat(fakeP);
    PCA_constrain(fakePMat, pca, PCA_BOX, 3);

    PCA_backProject_pts(fakeP, dataOut, pca);

    draw_body_pts(img, dataOut, Scalar(0,255,255));
    imshow("PCA", img);
    waitKey(0);

    ///Next step:
    ///Test template matching function


//    PCA_constrain_pts(dataIn, dataOut, pca);
//    draw_body_pts(img, dataOut, Scalar(0,255,255));
//    imshow("PCA_constrain", img);
//    waitKey(0);



//    for(srcVid.read(frame); srcVid.read(frame);)
//    {
//        do
//        {
//            if(!srcVid.read(frame))
//                continue;
//            cvtColor(frame, frame, CV_BGR2GRAY); //make frame grayscale
//            extract_fg(frame, bg, fgMask, 7, MORPH_ELLIPSE, true, true);
//        }while(!find_4_skydiver_blobs(fgMask, skydiverBlobs, skydiverBlobContours));
//
//        SkydiverBlob skydivers[4];
//
//        for(int i=0; i<4; i++)
//            skydivers[i].approx_parameters(skydiverBlobContours[i], skydiverBlobs.rows, skydiverBlobs.cols);
//
//
//        vector<vector<Point2f> > initialModelFit(4);
//        Mat skydiverBlobsParams(skydiverBlobs.size(), skydiverBlobs.type(), Scalar(0,0,0));
//
//        for(int i=0; i<4; i++) //For each skydiver blob
//        {
//            double initialScale = skydivers[i].scaleMetric/meanScaleMetric;
//            Point2f initialTranslation = skydivers[i].centroid - meanCentroid;
//            double initialRotation = skydivers[i].orientation - meanOrientation;
//
//            Mat initialModelFitMat = meanPointsMat.clone();
//            initialModelFitMat += Scalar(initialTranslation.x, initialTranslation.y);
//
//            initialModelFitMat.reshape(2).copyTo(initialModelFit[i]);
//
//            initialModelFit[i] = rotate_pts(initialModelFit[i], initialRotation, initialScale);
//
//            //Output
//            Mat params = skydivers[i].paramaters_image().clone();
//            Scalar colour(128);
//            draw_body_pts(params, initialModelFit[i], colour);
////            if(skydivers[i].flag)
//                skydiverBlobsParams += params;
////            draw_angle(skydiverBlobs, skydivers[i].centroid, skydivers[i].orientation);
//
//    //        stringstream numSS("");
//    //        numSS << i;
//    //        imwrite("blob_params_" + numSS.str() + ".jpg", params);
//
//        }
//        imshow("Window", skydiverBlobsParams);
//        waitKey(100);
//    }

   return 0;
}
