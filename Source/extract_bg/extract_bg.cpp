#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include "extract_bg.h"
#include "../mw_libCV.h"


#define FCC CV_FOURCC('P','I','M','1') //MPEG-1 codec compression
//#define FCC 0 //uncompressed

using namespace std;
using namespace cv;

int main()
{
    bool useStdDev = true;
    int minStdDevs = 2;
    int medianFilerSize = 5;
    int medianFilerShape = MORPH_RECT;

     VideoCapture srcVid(SRC_VID_PATH), dstVid;
    if(!srcVid.isOpened())
        cout << "Could not open input video for writing: " << SRC_VID_PATH << endl;
    VideoWriter outVid_fg;

    Mat frame, temp, bg;

    bg = imread(BG_IMG_PATH, CV_LOAD_IMAGE_GRAYSCALE);
    if(bg.data == NULL)
        cout << "Could not open background image: " << BG_IMG_PATH << endl;

    Size srcSize = Size((int)(srcVid.get(CV_CAP_PROP_FRAME_WIDTH) * SCALE_FACTOR), (int)(srcVid.get(CV_CAP_PROP_FRAME_HEIGHT)) * SCALE_FACTOR);

    outVid_fg.open("../../Data/src/vid/4-way_fs_dive-pool_fg.avi", FCC, srcVid.get(CV_CAP_PROP_FPS), srcSize, 0);

    if(!outVid_fg.isOpened())
        cout << "Could not open output video for writing" << endl;


    namedWindow("Frame", WINDOW_NORMAL);

    for(srcVid.read(frame); srcVid.read(frame);)
    {
        cvtColor(frame, frame, CV_BGR2GRAY);
        extract_fg(frame, bg, frame, 7, MORPH_RECT, true, true, 1);
        imshow("Frame", frame);
        outVid_fg << frame;
        waitKey(1);
    }

    return 0;
}
