#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "data_label.h"

#define SRC_VID_DIR (std::string)"../../Data/src/vid/"
#define SRC_IMG_DIR (std::string)"../../Data/src/img/"
#define OUT_DIR (std::string)"../../Data/out/point_samples/"

#define WHITE 255
#define BLACK 0

using namespace std;
using namespace cv;

int main()
{
    int nImgs = 10;
    string src_vid_name = "4-way_fs_dive-pool";
    VideoCapture srcVid(SRC_VID_DIR + src_vid_name + ".avi");
    Mat frame, mask, subimg;
    const int nFrames = srcVid.get(CV_CAP_PROP_FRAME_COUNT)/2;
    const int srcX = srcVid.get(CV_CAP_PROP_FRAME_WIDTH);
    const int srcY = srcVid.get(CV_CAP_PROP_FRAME_HEIGHT);
    const int framesToUse = 3;
    const Size sampleSize(40,40);
    const int videoFps = 2;

    string opPointDataPath(OUT_DIR + "data_points__" + src_vid_name + ".txt");
//    vector<vector<Mat> /*Size = 11*/> desc_pt_samples;
    VideoWriter sampleVids[11];
    for(int i=0; i<11; i++)
        sampleVids[i].open(OUT_DIR + get_point_name(i)+ "__" + src_vid_name + ".avi", 0, videoFps, sampleSize*2);
        //upscaled output video size as there is a minimum size for output videos

    stringstream frameText;

    namedWindow("Frame", CV_WINDOW_NORMAL);

    float angle;
    for(int i=framesToUse+1; i<nFrames; i++)
    {
        srcVid.read(frame);
        mask = frame.clone();
        mask = Scalar(BLACK);
        if((i%((int)floor((nFrames/(framesToUse)) + 0.5))) == framesToUse) //select frames to use
        {
            frameText.str(string());
            frameText << "Frame:" << i << "/" << nFrames;
            putText(frame, frameText.str(), Point(0,srcY),
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(WHITE));
            for(int j=0; j<4; j++)
            {
                get_desc_points(frame);
                if(desc_pts.size() == 11)
                {
                //0  Waist
                    get_subimgs(desc_pts[WAIST], desc_pts[NECK], sampleSize, frame, sampleVids[WAIST]);
                //1  Neck
                    get_subimgs(desc_pts[NECK], desc_pts[WAIST], sampleSize, frame, sampleVids[NECK]);
                //2  Head
                    get_subimgs(desc_pts[HEAD], desc_pts[NECK], sampleSize, frame, sampleVids[HEAD]);
                //3  Left hand
                    get_subimgs(desc_pts[LEFT_HAND], desc_pts[LEFT_ELBOW], sampleSize, frame, sampleVids[LEFT_HAND]);
                //4  Left elbow
                    get_subimgs(desc_pts[LEFT_ELBOW], desc_pts[NECK], sampleSize, frame, sampleVids[LEFT_ELBOW]);
                //5  Left knee
                    get_subimgs(desc_pts[LEFT_KNEE], desc_pts[WAIST], sampleSize, frame, sampleVids[LEFT_KNEE]);
                //6  Left foot
                    get_subimgs(desc_pts[LEFT_FOOT], desc_pts[LEFT_KNEE], sampleSize, frame, sampleVids[LEFT_FOOT]);
                //7  Right foot
                    get_subimgs(desc_pts[RIGHT_FOOT], desc_pts[RIGHT_KNEE], sampleSize, frame, sampleVids[RIGHT_FOOT]);
                //8  Right knee
                    get_subimgs(desc_pts[RIGHT_KNEE], desc_pts[WAIST], sampleSize, frame, sampleVids[RIGHT_KNEE]);
                //9  Right elbow
                    get_subimgs(desc_pts[RIGHT_ELBOW], desc_pts[NECK], sampleSize, frame, sampleVids[RIGHT_ELBOW]);
                //10 Right hand
                    get_subimgs(desc_pts[RIGHT_HAND], desc_pts[RIGHT_ELBOW], sampleSize, frame, sampleVids[RIGHT_HAND]);

                    imshow("Frame", frame);
                    write_points(opPointDataPath.c_str() , desc_pts, true);
                }
            }
            //imwrite(OUT_DIR + "data_label_samples.jpg", frame);
        }
    }

    return 0;
}
