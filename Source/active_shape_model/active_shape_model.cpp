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



void upscale_data(Mat& data)
{
    data *= 500;
    data += Scalar(300, 300);
}

void show_PCA_component_sliders(vector<Mat>& GPA_data, Mat& GPA_mean, int n_components, int component_max, int component_min, Size window_size = Size(1000,1000))
{
    Mat dataMatPCA = formatImagesForPCA(GPA_data);
    vector<Mat> meanMat_{GPA_mean};
    Mat meanMatPCA = formatImagesForPCA(meanMat_);

    namedWindow("PCA component sliders", WINDOW_AUTOSIZE);
    PCA pca(dataMatPCA, meanMatPCA, CV_PCA_DATA_AS_ROW, n_components);

    int initialVal = ((component_max - component_min) / 2);
    int maxVal = component_max - component_min;
    int component[n_components];

    for(int i=0; i<n_components; i++)
    {
        component[i] = initialVal;
        stringstream ss;
        ss << i;
        createTrackbar(string("C") + ss.str().c_str(), "PCA component sliders", &component[i], maxVal);
    }

    while(1)
    {
        Mat pcaShapeOp(window_size, CV_8UC3, Scalar(255, 255, 255));

        float descriptors[n_components];
        for(int i=0; i<n_components; i++)
            descriptors[i] = component[i] + component_min;

        Mat pcaShape = pca.backProject(Mat(1, n_components, CV_32F, &descriptors));

        vector<Point2f> pcaShapeVec;
        reformatImageFromPCA(pcaShape).copyTo(pcaShapeVec);

        Scalar colour(0, 0, 0);
        draw_body_pts(pcaShapeOp, pcaShapeVec, colour);

        stringstream ss;
        for(int i=0; i< n_components; i++)
        {
            ss << "C" << i << "=" << (int)descriptors[i];
            if(i+1 < n_components)
                ss << ", ";
        }

        putText(pcaShapeOp, ss.str().c_str(), Point(5, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(128, 128, 128));

        imshow("PCA component sliders", pcaShapeOp);
        waitKey(1);
    }
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


//PCA
    Mat dataMatPCA = formatImagesForPCA(dataMatGPA);
    vector<Mat> meanMat_{meanMat};
    Mat meanMatPCA = formatImagesForPCA(meanMat_);


//    int nComponents = 5;
    double retainedVariance = 0.9;
    PCA pca(dataMatPCA, meanMatPCA, CV_PCA_DATA_AS_ROW, retainedVariance);
    PCA_save(pca, "../../Data/out/PCA.yml");

    show_PCA_component_sliders(dataMatGPA, meanMat, 5, 100, -100);

//    namedWindow("pcaShapeOp", WINDOW_AUTOSIZE);

//    vector<vector<Point2f> > gpaDataVecPCA(gpaDataVec.size());
//    vector<Mat> gpaDataMatPCA;
//    Mat gpaDataPCAop(1000, 1000, CV_8UC3, Scalar(255, 255, 255));
//    for(int i=0; i<gpaDataVec.size(); i++)
//    {
//        Mat temp = pca.project(dataMatPCA.row(i));
//        cout << "nComponents = " << temp.cols << endl;
//        temp = pca.backProject(temp);
//        reformatImageFromPCA(temp).copyTo(gpaDataVecPCA[i]);
//        plot_pts(gpaDataPCAop, gpaDataVecPCA[i], colours[i]);
//    }
//
//    namedWindow("GPA shapes PCA", WINDOW_AUTOSIZE);
//    imshow("GPA shapes PCA", gpaDataPCAop);

    waitKey(0);
    return 0;
}



















