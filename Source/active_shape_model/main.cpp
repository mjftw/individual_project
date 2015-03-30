#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Procrustes.h"
#include "../mw_libCV.h"

using namespace std;
using namespace cv;

int main()
{
    vector<vector<Point2f> > data;
    load_data_pts("data_points.txt", data);
    vector<Mat> dataMat;
    vector<Scalar> colours;

    Mat dataOp(1000, 1000, CV_8UC3, Scalar(255, 255, 255));

    for(int i=0; i<data.size(); i++)
    {
        dataMat.push_back(Mat(data[i]));
        colours.push_back(Scalar(rand()%256, rand()%256, rand()%256));
        plot_pts(dataOp, data[i], colours[i]);
    }

    Mat meanMat;
    Procrustes proc;
    dataMat = proc.generalizedProcrustes(dataMat, meanMat);
    vector<vector<Point2f> > meanMatVec(1);
    meanMat.reshape(2).copyTo(meanMatVec[0]);

    write_data_pts("data_points_mean.txt", meanMatVec);

    for(int i=0; i<dataMat.size(); i++)
        dataMat[i].reshape(2).copyTo(data[i]);

    write_data_pts("data_points_PA.txt", data);

    Mat meanOp(1000, 1000, CV_8UC3, Scalar(255, 255, 255));

    vector<Point2f> meanShape;
    meanMat *= 500;
    meanMat += Scalar(300, 300);
    meanMat.reshape(2).copyTo(meanShape);

    plot_pts(meanOp, meanShape, colours[0]);

    Mat GPAOp(1000, 1000, CV_8UC3, Scalar(255, 255, 255));
    for(int i=0; i<dataMat.size(); i++)
    {
        vector<Point2f> tempShape;
        dataMat[i] *= 500;
        dataMat[i] += Scalar(300, 300);
        dataMat[i].reshape(2).copyTo(tempShape);
        plot_pts(GPAOp, tempShape, colours[i]);
    }
    namedWindow("data points", WINDOW_AUTOSIZE);
    imshow("data points", dataOp);
    namedWindow("mean shape", WINDOW_AUTOSIZE);
    imshow("mean shape", meanOp);
    namedWindow("GPA shapes", WINDOW_AUTOSIZE);
    imshow("GPA shapes", GPAOp);

    imwrite("data_points.jpg", dataOp);
    imwrite("mean_shape.jpg", meanOp);
    imwrite("GPA_shapes.jpg", GPAOp);

    waitKey(0);
    return 0;
}





















