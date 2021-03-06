#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

#define WHITE 255
#define BLACK 0

#define SCALE_FACTOR 1


void edge_detect(Mat& src, Mat& dst);
void find_skeleton(Mat& src, Mat& dst); //Simple morphological skeletonisation
void find_skeleton_connected(Mat& src, Mat& dst); //Medial Axis Transformation skeletonisation method, taken from Digital Image Processing book, pg650-653
void edge_detect(Mat& src, Mat& dst);
vector<vector<Point> > reduce_points(Mat& src, Mat& dst, float epsilon, float sizeTol = 0.3); //uses Ramer�Douglas�Peucker algorithm
void sub_vid_bg(VideoCapture& src, VideoCapture& dst);

void find_skeleton(Mat& src, Mat& dst)
{
    // Using morphological method (bad)
    Mat skel(src.size(), CV_8UC1, Scalar(BLACK));
    Mat temp(src.size(), CV_8UC1);
    Mat element = getStructuringElement(MORPH_CROSS, cv::Size(3, 3));
    double maximum = BLACK;

    do
    {
        morphologyEx(src, temp, cv::MORPH_OPEN, element);
        bitwise_not(temp, temp);
        bitwise_and(temp, src, temp);
        bitwise_or(temp, skel, skel);
        erode(src, src, element);       //src(erode)

        minMaxLoc(src, BLACK, &maximum);
    }while (maximum != BLACK);

    morphologyEx(src, src, cv::MORPH_CLOSE, element);

    dst = skel.clone();
    return;
}

void find_skeleton_connected(Mat& src, Mat& dst)
{
    /*  Medial Axis Transformation skeletonisation method, taken from Digital Image Processing book, pg650-653
        Algorithm steps:
        For each boundary pixel:
            1. test pixels neighbours
            p8 p1 p2  |
            p7 p0 p3  | +y
            p6 p5 p4  \/
            ---------->
                +x

            2. Count Np0, number of non-zero neighbours of p0
            3. Count Tp0, number of 0-1 transitions in sequence p2, p3 .... p7, p8

            4. Check first conditions, and mark for deletion any point that meets all conditions
                cA:     2 <= Np0 <= 6
                cB:     Tp1 = 1
                cC:     p1 . p3 . p5 = 0
                cD:     p3 . p5 . p7 = 0
        5. Delete any points marked for deletion

        For each remaining pixel:
            6. test pixels neighbours
            p8 p1 p2
            p7 p0 p3
            p6 p5 p4

            7. Check second conditions, and mark for deletion any point that meets all conditions
                cA:     2 <= Np0 <= 6
                cB:     Tp1 = 1
                cC_:    p1 . p3 . p7 = 0
                cD_:    p1 . p5 . p7 = 0
        8. Delete any points marked for deletion

        repeat until no points are deleted in an iteration
    */

    Mat skel = src.clone(); //Src MUST be a binary image
    Mat skel_prev(src.size(), CV_8UC1, Scalar(BLACK)), skel_diff(src.size(), CV_8UC1, Scalar(BLACK));
    Mat to_delete(src.size(), CV_8UC1, Scalar(WHITE)); //matrix or points to be deleted
    Mat edge(src.size(), CV_8UC1, Scalar(BLACK));
    double maximum = BLACK;

    bool p[9];  // flags showing if 8-neighbour points of p[0] are 0 or 1
    bool cA, cB, cC, cD, cC_, cD_; //flags showing if conditions are met
    int Tp0, Np0;
    do
    {
        skel_prev = skel.clone();
        edge_detect(skel, edge);
        //cannot test edge points, hence the -2
        for(int i=1; i<(src.rows-2); i++)    //x values
            for(int j=1; j<(src.cols-2); j++) //y values
            {
                if(edge.at<uchar>(i,j) == WHITE)
                {
                    Np0 = 0;
                    Tp0 = 0;

                    p[0] = skel.at<uchar>(i  , j  );
                    p[1] = skel.at<uchar>(i-1, j  );
                    p[2] = skel.at<uchar>(i-1, j+1);
                    p[3] = skel.at<uchar>(i  , j+1);
                    p[4] = skel.at<uchar>(i+1, j+1);
                    p[5] = skel.at<uchar>(i+1, j  );
                    p[6] = skel.at<uchar>(i+1, j-1);
                    p[7] = skel.at<uchar>(i  , j-1);
                    p[8] = skel.at<uchar>(i-1, j-1);

                    for(int k=1; k<9; k++)
                    {
                        if(p[k])
                            Np0++;
                        if((k!=1) && (p[k] == 1) && (p[k-1] == 0))
                            Tp0++;
                    }
                    if((p[1] == 1) && (p[8] == 0))
                        Tp0++;

                    cA = ((2 <= Np0) && (Np0 <= 6));
                    cB = (Tp0 == 1);
                    cC = ((p[1] & p[3] & p[5]) == 0);
                    cD = ((p[3] & p[5] & p[7]) == 0);

                    if(cA & cB & cC & cD)
                        to_delete.at<uchar>(i,j) = BLACK;
                }
            }

        bitwise_and(skel, to_delete, skel);
        edge_detect(skel, edge);

        for(int i=1; i<(src.rows-2); i++)    //y values
            for(int j=1; j<(src.cols-2); j++) //x values
            {
                if(edge.at<uchar>(i,j) == WHITE)
                {
                    Np0 = 0;
                    Tp0 = 0;

                    p[0] = skel.at<uchar>(i  , j  );
                    p[1] = skel.at<uchar>(i-1, j  );
                    p[2] = skel.at<uchar>(i-1, j+1);
                    p[3] = skel.at<uchar>(i  , j+1);
                    p[4] = skel.at<uchar>(i+1, j+1);
                    p[5] = skel.at<uchar>(i+1, j  );
                    p[6] = skel.at<uchar>(i+1, j-1);
                    p[7] = skel.at<uchar>(i  , j-1);
                    p[8] = skel.at<uchar>(i-1, j-1);

                    for(int k=1; k<9; k++)
                    {
                        if(p[k])
                            Np0++;
                        if((k!=1) && (p[k] == 1) && (p[k-1] == 0))
                            Tp0++;
                    }
                    if((p[1] == 1) && (p[8] == 0))
                        Tp0++;

                    cA = ((2 <= Np0) && (Np0 <= 6));
                    cB = (Tp0 == 1);
                    cC_ = ((p[1] & p[3] & p[7]) == 0);
                    cD_ = ((p[1] & p[5] & p[7]) == 0);

                    if(cA & cB & cC_ & cD_)
                        to_delete.at<uchar>(i,j) = BLACK;
                }
            }

        bitwise_and(skel, to_delete, skel);

        absdiff(skel, skel_prev, skel_diff);
        minMaxLoc(skel_diff, BLACK, &maximum);
    }while(maximum != BLACK);

    dst = skel.clone();
    return;
}

void edge_detect(Mat& src, Mat& dst)
{
    Mat edge(src.size(), CV_8UC1, Scalar(BLACK));
    Mat kernel = (Mat_<char>(3,3)  << -1, -1, -1,
                                      -1,  8, -1,
                                      -1, -1, -1); //laplacian edge detection
    filter2D(src, dst, -1, kernel);
    return;
}

vector<vector<Point> > reduce_points(Mat& src, Mat& dst, float epsilon, float sizeTol)
{
    /* Algorithm plan:
        1. Find points where Tp0 <=3
        2. Mask these bifurcation points to create separate polys
        3. Find contours
        4. Ramer�Douglas�Peucker algorithm on each contour, pseudocode used from wiki page
        5. combine contours
    */
    Mat temp = src.clone();
    Mat temp2(src.size(), CV_8UC1, Scalar(BLACK));

    vector<Point> splitPt;
    vector<vector<Point> > lineSegs, lineSegsReduced;
    bool p[9];
    int Tp0;
    int avgSegSize = 0;

    for(int i=1; i<(src.rows-2); i++)    //x values
        for(int j=1; j<(src.cols-2); j++) //y values
        {
            if(src.at<uchar>(i,j) == WHITE)
            {
                Tp0 = 0;

                p[0] = src.at<uchar>(i  , j  );
                p[1] = src.at<uchar>(i-1, j  );
                p[2] = src.at<uchar>(i-1, j+1);
                p[3] = src.at<uchar>(i  , j+1);
                p[4] = src.at<uchar>(i+1, j+1);
                p[5] = src.at<uchar>(i+1, j  );
                p[6] = src.at<uchar>(i+1, j-1);
                p[7] = src.at<uchar>(i  , j-1);
                p[8] = src.at<uchar>(i-1, j-1);

                for(int k=1; k<9; k++)
                    if((k!=1) && (p[k] == 1) && (p[k-1] == 0))
                        Tp0++;
                if((p[1] == 1) && (p[8] == 0))
                    Tp0++;

                if(Tp0 > 2)
                {
                    splitPt.push_back(Point(i,j));

                    temp.at<uchar>(i  , j  ) = BLACK;
                    temp.at<uchar>(i-1, j  ) = BLACK;
                    temp.at<uchar>(i-1, j+1) = BLACK;
                    temp.at<uchar>(i  , j+1) = BLACK;
                    temp.at<uchar>(i+1, j+1) = BLACK;
                    temp.at<uchar>(i+1, j  ) = BLACK;
                    temp.at<uchar>(i+1, j-1) = BLACK;
                    temp.at<uchar>(i  , j-1) = BLACK;
                    temp.at<uchar>(i-1, j-1) = BLACK;
                }
            }
        }

    findContours(temp, lineSegs, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);

    for(int i=0; i<lineSegs.size(); i++)
        avgSegSize += arcLength(lineSegs[i], false);
    avgSegSize /= lineSegs.size();

    for(int i=0; i<lineSegs.size(); i++)
    {
        if(arcLength(lineSegs[i], false) >= (avgSegSize * sizeTol))
        {
            lineSegsReduced.push_back(lineSegs[i]);
            approxPolyDP(lineSegs[i], lineSegsReduced[lineSegsReduced.size()-1], epsilon, true);
        }
    }


    drawContours(temp2, lineSegsReduced, -1, Scalar(WHITE));

    for(int i=0; i<splitPt.size(); i++)
        circle(temp2, Point(splitPt[i].y, splitPt[i].x), 3, Scalar(WHITE), 2);

    dst = temp2.clone();
    return lineSegsReduced;
}

void sub_vid_bg(VideoCapture& src, VideoCapture& dst)
{
    BackgroundSubtractorMOG mog(300,10,0.9);
    Mat frame, fg_mask, fg_mask_opn;
    int morph_size = 3;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*morph_size+1, 2*morph_size+1), Point(morph_size, morph_size));

    namedWindow("frame", WINDOW_NORMAL);
    namedWindow("fg_mask", WINDOW_NORMAL);
    namedWindow("fg_mask_opn", WINDOW_NORMAL);

    for(src.read(frame); src.read(frame);)
    {
        imshow("frame", frame);
        mog(frame, fg_mask);
        threshold(fg_mask, fg_mask, 1, WHITE, THRESH_BINARY);
        morphologyEx(fg_mask, fg_mask_opn, MORPH_OPEN, element);

        imshow("fg_mask", fg_mask);
        imshow("fg_mask_opn", fg_mask_opn);

        waitKey(1);
    }

    return;
}

void getHist(uchar* prev_n_pixels, int arry_length, unsigned int* hist)
{
    for(int i=0; i<255; i++)
        hist[i] = 0;

    for(int i=0; i<arry_length; i++)
        hist[prev_n_pixels[i]]++;
}

void extract_bg(VideoCapture& srcVid, Mat& bg, int nth_frame)
{
    cout << "Calculating background image..." << endl;
    Mat frame;

    const int rows = srcVid.get(CV_CAP_PROP_FRAME_HEIGHT) * SCALE_FACTOR;
    const int cols = srcVid.get(CV_CAP_PROP_FRAME_WIDTH) * SCALE_FACTOR;
    const int n_frames = srcVid.get(CV_CAP_PROP_FRAME_COUNT);
    const int n_frames_used_q = div(n_frames, nth_frame).quot;
    const int n_frames_used_r = div(n_frames, nth_frame).rem;

    cout << "Using maximum " << n_frames_used_q << " of " << n_frames << " frames at " << cols << "x" << rows << " resolution" << endl;

    uchar*** prev_frames = new uchar**[rows];
    for(int i=0; i<rows; i++)
    {
        prev_frames[i] = new uchar*[cols];
        for(int j=0; j<cols; j++)
            prev_frames[i][j] = new uchar[n_frames_used_q];
    }

    int k=0;
    for(srcVid.read(frame); srcVid.read(frame); k++)
    {
        cvtColor(frame, frame, CV_BGR2GRAY); //make frame grayscale
        resize(frame, frame, Size(cols,rows), 0 ,0); //resize frame

        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                prev_frames[i][j][k] = frame.at<uchar>(i,j);
        srcVid.set(CV_CAP_PROP_POS_FRAMES, srcVid.get(CV_CAP_PROP_POS_FRAMES) + n_frames_used_r);
    }

    int N = k;
    cout << "Calculating background image using " << N << " frames..." << endl;

    Mat median(rows, cols, CV_8UC1, Scalar(255));

    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
        {///implementation of algorithm "medhist_bnd(D)" from paper "Speed Up Temporal Median Filter for Background Subtraction"
        ///Mao-Hsiung Hung, Jeng-Shyang Pan, Chaur-Heh Hsieh (2010)

            int O_mid = (N%2 == 0)? (N/2):(N-1)/2;
            int csum = 0;
            unsigned int hn[256];
            getHist(prev_frames[i][j], N, hn);

            int k;
            for(k=0; k<255; k++)
            {
                if(hn[k]>0)
                    csum += hn[k];
                if(csum>=O_mid)
                    break;
            }
            median.at<uchar>(i,j) = k;
        }


    bg = median.clone();

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
            delete prev_frames[i][j];
        delete prev_frames[i];
    }
    delete prev_frames;
}
