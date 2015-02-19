#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "libs.h"

using namespace std;
using namespace cv;

int main()
{

    string src_img_path = "Data/Pictures/bp9_mask.tif";
    string src_vid_path = "../Data/src/vid/4-way_fs_dive-pool.avi";
    Mat src_img, dst;
    VideoCapture srcVid(src_vid_path), dstVid;

    Mat frame, fg_mask;

    int i=0;
    for(srcVid.read(frame); srcVid.read(frame); i++)
    {
        if(i%10 == 0)
            extract_fg(frame, fg_mask, 20);
        waitKey(1);
    }
    namedWindow("bg", WINDOW_NORMAL);
    imshow("bg", fg_mask);


    //sub_vid_bg(srcVid, dstVid);
    //cout << "After bg sub." << endl;

//      Mat bg_img = imread();
//    compute_bg_img(srcVid, bg_img, 100);
//    namedWindow("bg", WINDOW_NORMAL);
//    imshow("bg", bg_img);

//    src_img = imread(src_img_path, CV_LOAD_IMAGE_GRAYSCALE);
//    if(src_img.empty())
//    {
//        cout << "Error loading source image" << endl;
//        return -1;
//    }
//
//    float sf = 0.25;    //scale factor
//    resize(src_img, src_img, Size(0,0), sf, sf);
//
//    threshold(src_img, src_img, 1, WHITE, THRESH_BINARY);
//    Mat element = getStructuringElement(MORPH_CROSS, cv::Size(3, 3)); //only required because image compression creates noisy edges & algorithm is very susceptible to noise
//    morphologyEx(src_img, src_img, cv::MORPH_OPEN, element);
//
//    namedWindow("src_img", WINDOW_NORMAL);
//    imshow("src_img", src_img);
//
//    find_skeleton_connected(src_img, dst);
//    find_skeleton(src_img, dst);
//
//    float epsilon = 30, sizeTol = 0.3;
//    reduce_points(dst, dst, epsilon, sizeTol);
//
//    namedWindow("Final", WINDOW_NORMAL);
//    imshow("Final", dst);

    waitKey(0);
    return 0;
}
