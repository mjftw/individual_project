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

    //--------------LOAD DATA--------------
    VideoCapture srcVid(SRC_VID_PATH);
    if(!srcVid.isOpened())
    {
        cerr << "ERROR: Cannot open video file " << SRC_VID_PATH << endl;
        exit(EXIT_FAILURE);
    }
    else
        cout << "Source video loaded" << endl;

    Mat bg = imread(BG_IMG_PATH, CV_LOAD_IMAGE_GRAYSCALE);
    if(bg.data == NULL)
    {
        cerr << "ERROR: Cannot open background image " << BG_IMG_PATH << endl;
        exit(EXIT_FAILURE);
    }
    else
        cout << "Background image loaded" << endl;

    vector<Mat> templates;
    {
    bool templatesLoaded = true;
    for(int i=0; i<11; i++)
    {
        stringstream ss("");
        ss << TEMPLATES_PATH << TEMPLATES_NAME << "_" << get_point_name(i) << "_" << ".bmp";
        Mat templ = imread(ss.str(), CV_LOAD_IMAGE_GRAYSCALE);
        if(templ.data == NULL)
        {
            cerr << "ERROR: Cannot open template " << ss.str() << endl;
            templatesLoaded = false;
        }
        templates.push_back(templ);
    }
    if(!templatesLoaded)
        exit(EXIT_FAILURE);
    else
        cout << "Template images loaded" << endl;
    }

    vector<vector<Point2f> > meanPointsVec;
    if(!load_data_pts(PROCRUSTES_MEAN_DATA_PATH, meanPointsVec))
    {
        cerr << "ERROR: Cannot open mean data points file " << PROCRUSTES_MEAN_DATA_PATH << endl;
        exit(EXIT_FAILURE);
    }
    else
        cout << "Procrustes mean data loaded" << endl;
    vector<Point2f> meanPoints = meanPointsVec[0];
    Mat meanPointsMat(meanPoints);

    PCA pca;
    if(!PCA_load(pca, PCA_DATA_PATH))
        cerr << "ERROR: Cannot open PCA model data file " << PCA_DATA_PATH << endl;
    else
        cout << "PCA model data loaded" << endl;


    //--------------JUST FOR TESTING--------------

    namedWindow("Test");
    Mat testImg(bg.size(), bg.type(), Scalar(0,0,0));

    //--------------INITIALISATION--------------

    double meanScaleMetric = get_scale_metric(meanPointsMat);
    Point2f meanCentroid = get_vec_centroid(meanPoints);
    double meanOrientation = get_major_axis(meanPointsMat);

    Mat skydiverBlobs;
    Mat frame, fgMask;
    vector<vector<Point> > skydiverBlobContours;
    do
    {
        if(!srcVid.read(frame))
        {
            cerr << "ERROR: Initialisation failed, could not find skydivers" << endl;
            exit(EXIT_FAILURE);
        }
        cvtColor(frame, frame, CV_BGR2GRAY); //make frame grayscale
        extract_fg(frame, bg, fgMask, 7, MORPH_ELLIPSE, true, true);
    }while(!find_4_skydiver_blobs(fgMask, skydiverBlobs, skydiverBlobContours));

    SkydiverBlob skydivers[4];

    for(int i=0; i<4; i++)
        skydivers[i].approx_parameters(skydiverBlobContours[i], skydiverBlobs.rows, skydiverBlobs.cols);

    vector<vector<Point2f> > initialModel(4);
    Mat skydiverBlobsParams(skydiverBlobs.size(), skydiverBlobs.type(), Scalar(0,0,0));

    for(int i=0; i<4; i++) //Fit procrustes mean to skydiver blobs
    {
        double initialScale = skydivers[i].scaleMetric/meanScaleMetric;
        Point2f initialTranslation = skydivers[i].centroid - meanCentroid;
        double initialRotation = skydivers[i].orientation - meanOrientation;

        Mat initialModelFitMat = meanPointsMat.clone();
        initialModelFitMat += Scalar(initialTranslation.x, initialTranslation.y);

        initialModelFitMat.reshape(2).copyTo(initialModel[i]);

        initialModel[i] = rotate_pts(initialModel[i], initialRotation, initialScale);

        PCA_constrain_pts(initialModel[i], initialModel[i], pca);

    }
    imshow("Test", testImg);
    waitKey(0);






//    Mat contourImg;
//    overlay_contours(skydiverBlobs, contourImg, skydiverBlobContours);
//    imshow("Window", skydiverBlobs);
//    waitKey(0);



//    namedWindow("PCA", WINDOW_NORMAL);
//    Mat img(bg.size(), CV_8UC3, Scalar(0,0,0));

//    vector<double> fakeP = {-3.815214, -0.671381, -0.262617, 0.594221, 0.0790387};
//    vector<Point2f> dataOut;
//
//    Mat fakePMat(fakeP);
//    PCA_constrain(fakePMat, pca, PCA_BOX, 3);
//
//    PCA_backProject_pts(fakeP, dataOut, pca);
//
//    draw_body_pts(img, dataOut, Scalar(0,255,255));
//    imshow("PCA", img);
//    waitKey(0);

    ///Next step:
    ///Test template matching function

//    Procrustes proc;

//    PCA_constrain_pts(dataIn, dataOut, pca);
//    draw_body_pts(img, dataon lOut, Scalar(0,255,255));
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
//        vector<vector<Point2f> > initialModel(4);
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
//            initialModelFitMat.reshape(2).copyTo(initialModel[i]);
//
//            initialModel[i] = rotate_pts(initialModel[i], initialRotation, initialScale);
//
//
//            //Output
//            Mat params = skydivers[i].paramaters_image().clone();
//            draw_body_pts(params, initialModel[i], Scalar(128));
//
//            skydiverBlobsParams += params;
//
//            vector<Point2f> nextPts;
//            //template matching
//            for(int j=0; j<11; j++)
//            {
//                double matchAmount = 0;
//                nextPts.push_back(template_match_point(frame, templates[j], 30, initialModel[i], j, CV_TM_SQDIFF, &matchAmount));
//                cout << "match amount: " << matchAmount << endl;
//            }
//            cout << "curr pts: " << Mat(initialModel[i]) <<  endl;
//            cout << "next pts: " << Mat(nextPts) <<  endl;
//
//
//            draw_body_pts(params, nextPts, Scalar(200));
//            imshow("Window", skydiverBlobsParams);
//
//
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
