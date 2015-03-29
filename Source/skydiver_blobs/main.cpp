#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "libs.h"
#include "mw_libCV.h"
#include "skydiverblob.h"

#define SRC_VID_DIR (std::string)"../../Data/src/vid/"
#define SRC_IMG_DIR (std::string)"../../Data/src/img/"
#define OUT_DIR (std::string)"../../Data/out/"

#define FCC CV_FOURCC('P','I','M','1') //MPEG-1 codec compression
//#define FCC 0 //uncompressed

using namespace std;
using namespace cv;

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
    waitKey(0);

    Mat skydiverBlobs[4];
    SkydiverBlob skydivers[4];

    for(int i=0; i<4; i++)
        skydivers[i].approx_parameters(skydiverBlobContours[i], src.rows, src.cols);

    for(int i=0; i<4; i++) //For each skydiver blob
    {
        cout << "Skydiver " << i << ": orientation = " << skydivers[i].orientation << " degrees, centroid = (" << skydivers[i].centroid.x << ", " << skydivers[i].centroid.y << ")" << endl;
        Mat params = skydivers[i].paramaters_image();
        imshow("Window", params);
        waitKey(0);
    }

   return 0;
}








