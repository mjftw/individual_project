#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../mw_libCV.h"

#define SRC_VID_DIR (std::string)"../../Data/src/vid/"
#define SRC_IMG_DIR (std::string)"../../Data/src/img/"
#define OUT_DIR (std::string)"../../Data/out/point_samples/"

#define WHITE 255
#define BLACK 0

using namespace std;
using namespace cv;

typedef struct
{
    vector<Point2f>* data_prt;
    Mat* img;
    Mat* frame;
    bool* skipFrame;
}DataStructType;

void call_back(int event, int x, int y, int flags, void* data_struct)
{
    DataStructType* data = static_cast<DataStructType*>(data_struct);

    if(event == EVENT_LBUTTONDOWN)
    {
        if(data->data_prt->size() < 11)
        {
            cout << get_point_name(data->data_prt->size()) << ": (" << x << ", " << y << "), " << get_point_name(data->data_prt->size()) << " next" << endl;
            data->data_prt->push_back(Point2f(x, y));
        }
        if(data->data_prt->size() == 11)
            cout << "All 11 points collected, space to continue or right click to undo"<< endl;
    }
    else if(event == EVENT_RBUTTONDOWN)
    {
        if(data->data_prt->size() > 0)
        {
            cout << "Last point removed" << endl;
            data->data_prt->pop_back();
        }
    }
    else if(event == EVENT_MBUTTONDOWN)
    {
        cout << "Press space to skip frame" << endl;
        *data->skipFrame = true;
    }

    *data->img = data->frame->clone();
    plot_pts(*data->img, *data->data_prt, Scalar(0, 255, 0));
    imshow("Frame", *data->img);

    return;
}

int main()
{

    Mat bg = imread(BG_IMG_PATH);
    cvtColor(bg, bg, CV_BGR2GRAY);

    VideoCapture srcVid(SRC_VID_PATH);
    const int nFrames = srcVid.get(CV_CAP_PROP_FRAME_COUNT)/2;

    const int framesToUse = 15;

    namedWindow("Frame", CV_WINDOW_NORMAL);

    vector<vector<Point2f> > landmarks;
    vector<Mat> frames;
    Mat frame, frameMask;

    setMouseCallback("Frame", call_back, &landmarks.back());

    cout << "Left mouse button: Add point" << endl;
    cout << "Left mouse button: Remove point" << endl;
    cout << "Middle mouse button: Skip frame" << endl << endl;

    bool skipFrame;
    for(int i=framesToUse; i<nFrames; i++)
    {
        srcVid.read(frame);
        if((i%((int)floor((nFrames/(framesToUse)) + 0.5))) == framesToUse) //select frames to use
        {
            cout << "Loading next frame" << endl;


            Mat frameAnnotated = frame.clone();
            Mat img = frame.clone();
            cvtColor(frame, frameMask, CV_BGR2GRAY);
            extract_fg(frameMask, bg, frameMask, 7, MORPH_RECT, true, true, 1);

            for(int j=0; j<4; j++)
            {
                landmarks.resize(landmarks.size()+1);

                imshow("Frame", img);

                skipFrame = false;
                DataStructType dataStruct = {&landmarks.back(), &img, &frameAnnotated, &skipFrame};
                setMouseCallback("Frame", call_back, &dataStruct);
                do
                {
                    waitKey(0);
                    if(skipFrame)
                    {
                        if(landmarks.size()%4 != 1)
                        {
                            cout << "Cannot skip frame, landmark vector(s) collected on frame" << endl;
                            skipFrame = false;
                        }
                        else
                        {
                            landmarks.pop_back();
                            break;
                        }
                    }
                    else if(landmarks.back().size() < 11)
                        cout << "Looks like you missed a point" << endl;

                }while(landmarks.back().size() < 11);
                if(skipFrame)
                    break;
                draw_body_pts(frameAnnotated, landmarks.back(), Scalar(255, 255, 0));
            }
            if(!skipFrame)
            {
                stringstream ss("");
                ss << (string)LANDMARKS_DIR + LANDMARKS_FRAMES_FILENAME << landmarks.size()/4 -1;
                imwrite((ss.str() + ".bmp").c_str(), frame);

                imwrite((ss.str() + "_bin.bmp").c_str(), frameMask);
            }
        }
    }
    setMouseCallback("Frame", 0);

    cout << landmarks.size() << " landmark vectors collected." << endl;
    cout << "Press space to save at " << LANDMARKS_FILENAME << endl;
    waitKey(0);

    if(!write_data_pts((string)LANDMARKS_DIR + LANDMARKS_FILENAME, landmarks))
        cout << "ERROR: Could not open " << LANDMARKS_FILENAME << " for writing" << endl;

    return 0;
}
