#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void get_desc_points(Mat& src);
string get_point_name(int pt);
void call_back(int event, int x, int y, int flags, void* userdata);
bool write_points(string path, bool amend);
void draw_points(Mat& src, Mat& dst);

vector<Point> desc_pts;
Mat SRCCLONE;

#define WHITE 255
#define BLACK 0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum{WAIST=0,NECK=1,HEAD=2,LEFT_HAND=3,LEFT_ELBOW=4,LEFT_KNEE=5, LEFT_FOOT=6,RIGHT_FOOT=7,RIGHT_KNEE=8,RIGHT_ELBOW=9,RIGHT_HAND=10};

inline double to_degrees(double radians)
{
    return radians * (180.0f / M_PI);
}
inline double to_radians(double degrees)
{
    return degrees * (M_PI / 180.0f);
}
inline double get_angle(Point pt1, Point pt2)
{
    double deltaX = pt2.x - pt1.x;
    double deltaY = pt2.y - pt1.y;
    return to_degrees(atan2(deltaY, deltaX));
}

void get_desc_points(Mat& src)
{
    SRCCLONE = src.clone();
    desc_pts.clear();

    namedWindow("Data Labelling", WINDOW_NORMAL);

    setMouseCallback("Data Labelling", call_back, NULL);

    imshow("Data Labelling", src);
        cout << "First point: " << get_point_name(0) << endl;
        waitKey(0); //wait for callback function to collect points

        if(desc_pts.size() < 11)
        {
            cout << "Insufficient number of points collected. Continuing..." << endl;
            waitKey(0);
        }

    return;
}

void call_back(int event, int x, int y, int flags, void* userdata)
{
    Mat dst;
    if(event == EVENT_LBUTTONDOWN)
    {
        if(desc_pts.size() < 11)
        {
            cout << '\t' << get_point_name(desc_pts.size()) << " (" << desc_pts.size() << ") : (" << x << ", " << y << "). " << get_point_name(desc_pts.size() + 1) << " point next." << endl;
            desc_pts.push_back(Point(x,y));
            if(desc_pts.size() == 11)
                cout << "All 11 points collected, press space finish." << endl;
        }
        else
        {
            cout << '\t' << "All 11 points collected, right click to remove last point." << endl;
        }
        draw_points(SRCCLONE, dst);
        imshow("Data Labelling", dst);
    }
    else if(event == EVENT_RBUTTONDOWN)
    {
        if(desc_pts.size() > 0)
        {
            cout << '\t' << get_point_name(desc_pts.size() - 1) << " point (" << (desc_pts.size() - 1) << ") removed." << endl;
            desc_pts.pop_back();
        }
        draw_points(SRCCLONE, dst);
        imshow("Data Labelling", dst);
    }

    return;
}

void draw_points(Mat& src, Mat& dst)
{
    dst = src.clone();
    for(int i=0; i<desc_pts.size(); i++)
        circle(dst, desc_pts[i], 3, Scalar(0,255,255));
}

string get_point_name(int pt)
{
    switch(pt)
    {
        case WAIST:
            return "waist";
        break;
        case NECK:
            return "neck";
        break;
        case HEAD:
            return "head";
        break;
        case LEFT_HAND:
            return "left_hand";
        break;
        case LEFT_ELBOW:
            return "left_elbow";
        break;
        case LEFT_KNEE:
            return "left_knee";
        break;
        case LEFT_FOOT:
            return "left_foot";
        break;
        case RIGHT_FOOT:
            return "right_foot";
        break;
        case RIGHT_KNEE:
            return "right_knee";
        break;
        case RIGHT_ELBOW:
            return "right_elbow";
        break;
        case RIGHT_HAND:
            return "right_hand";
        break;
        default:
            return "ERROR";
        break;
    }
}

bool write_points(string path, vector<Point> pts, bool append) //NOT WORKING PROPERLY
{
    ofstream dataFile;
    dataFile.open(path.c_str(), ios::out | (append? ios::app : ios::trunc));

    if(!dataFile.is_open())
    {
        cout << "Could not open output file for writing.";
        return 1;
    }
    else
        cout << "File opened at: " << path << endl;

    for(int i=0; i<pts.size(); i++)
    {
        dataFile << pts[i].y << ' ' << pts[i].x;
        if(i < pts.size()-1)
            dataFile << ' ';
        if(i == pts.size()-1)
            dataFile << endl;
    }

    dataFile.close();
    return 0;
}

float get_subimgs(Point& pt1, Point& pt2, Size rectSize, Mat& frame, VideoWriter& dstVid)
{
    Mat rotMat, bounding, rotated, cropped;
    float angle = get_angle(pt1, pt2);
    RotatedRect rotRect(pt1, rectSize, angle);

    rotMat = getRotationMatrix2D(rotRect.center, angle-90, 1.0);
    warpAffine(frame, rotated, rotMat, frame.size(), INTER_CUBIC);
    getRectSubPix(rotated, rectSize, rotRect.center, cropped);

    resize(cropped, cropped, Size(), 2, 2, INTER_CUBIC); // upscale 2x
    dstVid << cropped;

//// Uncomment in order to see the rotated rectangles on the frames
//    Point2f rectVerts[4];
//    rotRect.points(rectVerts);
//    for(int i=0; i<4; i++)
//        line(frame, rectVerts[i], rectVerts[(i+1)%4], Scalar(0,255,0));

    return angle;
}





