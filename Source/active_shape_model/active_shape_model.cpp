#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../../external_libs/Procrustes/Procrustes.h"
#include "../mw_libCV.h"

using namespace std;
using namespace cv;

///Taken from OpenCV PCA example program:
///opencv_source_code/samples/cpp/pca.cpp
static Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}

Mat reformatImageFromPCA(Mat& img, int channels = 1, int rows = 11)
{
    vector<double> colVec;
    vector<Point2f> pts;

    img.reshape(channels, rows);
    img.copyTo(colVec);

    for(int i=0; i<colVec.size()/2; i++)
        pts.push_back(Point2f(colVec[2*i], colVec[2*i +1]));

    return Mat(pts);
}


void upscale_data(Mat& data)
{
    data *= 500;
    data += Scalar(300, 300);
}

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
        cout << dataMat[i] << endl;
        colours.push_back(Scalar(rand()%256, rand()%256, rand()%256));
        plot_pts(dataOp, data[i], colours[i]);
    }

    Mat meanMat;
    Procrustes proc;
    vector<Mat> dataMatGPA = proc.generalizedProcrustes(dataMat, meanMat);
    vector<vector<Point2f> > meanMatVec(1);
    upscale_data(meanMat);
    meanMat.reshape(2).copyTo(meanMatVec[0]);

    write_data_pts("data_points_mean.txt", meanMatVec);

    for(int i=0; i<dataMatGPA.size(); i++)
    {
        upscale_data(dataMatGPA[i]);
        dataMatGPA[i].reshape(2).copyTo(data[i]);
    }


    write_data_pts("data_points_PA.txt", data);

    Mat meanOp(1000, 1000, CV_8UC3, Scalar(255, 255, 255));

    vector<Point2f> meanShape;
    meanMat.reshape(2).copyTo(meanShape);

    plot_pts(meanOp, meanShape, colours[0]);

    vector<vector<Point2f> > gpaDataVec(dataMatGPA.size());
    Mat GPAOp(1000, 1000, CV_8UC3, Scalar(255, 255, 255));
    for(int i=0; i<dataMatGPA.size(); i++)
    {
        Mat temp;
        dataMatGPA[i].reshape(2).copyTo(gpaDataVec[i]);
        plot_pts(GPAOp, gpaDataVec[i], colours[i]);
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

    Mat dataMatPCA = formatImagesForPCA(dataMatGPA);
    vector<Mat> meanMat_{meanMat};
    Mat meanMatPCA = formatImagesForPCA(meanMat_);

    double retainedVariance = 0.9;
    PCA pca(dataMatPCA, meanMatPCA, CV_PCA_DATA_AS_ROW, retainedVariance);

    vector<vector<Point2f> > gpaDataVecPCA(gpaDataVec.size());
    vector<Mat> gpaDataMatPCA;
    Mat gpaDataPCAop(1000, 1000, CV_8UC3, Scalar(255, 255, 255));
    for(int i=0; i<gpaDataVec.size(); i++)
    {
        Mat temp = pca.backProject(pca.project(dataMatPCA.row(i)));
        reformatImageFromPCA(temp).copyTo(gpaDataVecPCA[i]);
        plot_pts(gpaDataPCAop, gpaDataVecPCA[i], colours[i]);
    }


    namedWindow("GPA shapes PCA", WINDOW_AUTOSIZE);
    imshow("GPA shapes PCA", gpaDataPCAop);

    waitKey(0);
    return 0;
}



















