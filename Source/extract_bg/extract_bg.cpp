#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "extract_bg.h"
#include "../mw_libCV.h"

#define SRC_VID_DIR (std::string)"../../Data/src/vid/"
#define SRC_IMG_DIR (std::string)"../../Data/src/img/"
#define OUT_DIR (std::string)"../../Data/out/"

#define FCC CV_FOURCC('P','I','M','1') //MPEG-1 codec compression
//#define FCC 0 //uncompressed

using namespace std;
using namespace cv;

int main()
{

    //string src_img_path = "Data/Pictures/bp9_mask.tif";
    string bg_img_path = SRC_IMG_DIR + "tunnel-background.png";
//    string src_vid_name = "alphabet_dive";
    string src_vid_name = "4-way_fs_dive-pool";

    Mat src_img, dst;
    VideoCapture srcVid(SRC_VID_DIR + src_vid_name + ".avi"), dstVid;
    VideoWriter outVid_fg, outVid_skel, outVid_fgbgDiff;

    Mat frame, temp, bg;

    //extract_bg(srcVid, bg, 10); //use every 10th frame
    bg = imread(bg_img_path, CV_LOAD_IMAGE_GRAYSCALE);

    namedWindow("background image", WINDOW_NORMAL);
    imshow("background image", bg);
    imwrite(OUT_DIR + src_vid_name + "_bg.png", bg);

    srcVid.set(CV_CAP_PROP_POS_AVI_RATIO, 0); //reset srcVid

    Size srcSize = Size((int)(srcVid.get(CV_CAP_PROP_FRAME_WIDTH) * SCALE_FACTOR), (int)(srcVid.get(CV_CAP_PROP_FRAME_HEIGHT)) * SCALE_FACTOR);

    outVid_fg.open(OUT_DIR + src_vid_name + "_fg.avi", FCC, srcVid.get(CV_CAP_PROP_FPS), srcSize, 0);
    outVid_skel.open(OUT_DIR + src_vid_name + "_skel.avi", FCC, srcVid.get(CV_CAP_PROP_FPS), srcSize, 0);
    outVid_fgbgDiff.open(OUT_DIR + src_vid_name + "_fgbgDiff.avi", FCC, srcVid.get(CV_CAP_PROP_FPS), srcSize, 0);

    if((!outVid_fg.isOpened()) || (!outVid_skel.isOpened()) || (!outVid_fgbgDiff.isOpened()))
        cerr << endl << "Could not open output video(s) for writing." << endl;
    else
        cout << endl << "Writing videos..." << endl;

    namedWindow("Frame", WINDOW_NORMAL);

    for(srcVid.read(frame); srcVid.read(frame);)
    {
        cvtColor(frame, frame, CV_BGR2GRAY); //make frame grayscale
        resize(frame, frame, Size(0,0), SCALE_FACTOR, SCALE_FACTOR); //resize frame
        temp = cv::abs(frame - bg);
//        outVid_fgbgDiff << temp;

        threshold(temp, temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU); //Adaptive thresholding

        median_filter_binary(temp, temp, 5, MORPH_CROSS);

        imshow("Frame", temp);
        outVid_fg << temp;
        imwrite("frame.png", temp);
        //outVid_skel << temp;

        waitKey(1);
    }



//    sub_vid_bg(srcVid, dstVid);
//    cout << "After bg sub." << endl;

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
