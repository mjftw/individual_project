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

bool CLICKED = false;
Point BUT_PT1;
Point BUT_PT2;
Point BUT_PT3;
Point BUT_PT4;

void upscale_data(Mat& data)
{
    data *= 500;
    data += Scalar(300, 300);
}

typedef struct
{
    vector<int>* components;
    PCA* pca;
    int n_constraints;
    int sliderMin;
    float nStdDev;
}ButtonData;

void PCA_constrain_arr(ButtonData* btnData)
{
    vector<float> components(btnData->n_constraints);
    for(int i=0; i<btnData->n_constraints; i++)
    {
        components[i] = float(btnData->components->at(i) + btnData->sliderMin);
        Mat compMat(components);

        PCA_constrain(compMat, *btnData->pca, PCA_BOX, btnData->nStdDev);
        btnData->components->at(i) = floor(components[i] + 0.5) - btnData->sliderMin;

        stringstream ss("");
        ss << i;
        setTrackbarPos(string("C") + ss.str().c_str(), "PCA component sliders", btnData->components->at(i));
    }
}

void update_sliders(int, void* Data)
{
    ButtonData* data = static_cast<ButtonData*>(Data);
    PCA_constrain_arr(data);
}

void click_button(int event, int x, int y, int flags, void* Data)
{
    ButtonData* data = static_cast<ButtonData*>(Data);

    if(event == EVENT_LBUTTONDOWN)
    {
        if((x>=BUT_PT1.x) && (x<=BUT_PT2.x) && (y>=BUT_PT3.y) && (y<=BUT_PT2.y))
        {
            CLICKED = true;
            PCA_constrain_arr(data);
        }

    }
    else if(event == EVENT_LBUTTONUP)
        CLICKED = false;

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
    vector<int> component(n_components);

    float nStdDevs = 3;
    int nStdDevsx1k = nStdDevs*1000;

    for(int i=0; i<n_components; i++)
    {
        component[i] = initialVal;
        stringstream ss;
        ss << i;
        createTrackbar(string("C") + ss.str().c_str(), "PCA component sliders", &component[i], maxVal);
    }

        ButtonData btnData;
        btnData.components = &component;
        btnData.n_constraints = n_components;
        btnData.pca = &pca;
        btnData.sliderMin = component_min;
        setMouseCallback("PCA component sliders", click_button, (void*)&btnData);
        createTrackbar("stdDevs", "PCA component sliders", &nStdDevsx1k, 5000, update_sliders, (void*)&btnData);


        BUT_PT1 = Point(window_size.width - 155, 35);
        BUT_PT2 = Point(window_size.width - 25, 35);
        BUT_PT3 = Point(window_size.width - 25, 10);
        BUT_PT4 = Point(window_size.width - 155, 10);
    while(1)
    {
        Mat pcaShapeOp(window_size, CV_8UC3, Scalar(255, 255, 255));

        btnData.nStdDev = nStdDevs;
        nStdDevs = float(nStdDevsx1k)/1000.0;

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
            ss << "C" << i << "=" << (int)descriptors[i] << ", ";
        ss << "n std devs=" << nStdDevs;

        putText(pcaShapeOp, ss.str().c_str(), Point(5, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(128, 128, 128));

        if(CLICKED)
        {
            putText(pcaShapeOp, "Constrain", Point(pcaShapeOp.cols - 149,31 ), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(128, 128, 128));
            line(pcaShapeOp, Point(BUT_PT1.x+1, BUT_PT1.y+1), Point(BUT_PT2.x+1, BUT_PT2.y+1), Scalar(128, 128, 128), 1);
            line(pcaShapeOp, Point(BUT_PT2.x+1, BUT_PT2.y+1), Point(BUT_PT3.x+1, BUT_PT3.y+1), Scalar(128, 128, 128), 1);
            line(pcaShapeOp, Point(BUT_PT3.x+1, BUT_PT3.y+1), Point(BUT_PT4.x+1, BUT_PT4.y+1), Scalar(128, 128, 128), 1);
            line(pcaShapeOp, Point(BUT_PT4.x+1, BUT_PT4.y+1), Point(BUT_PT1.x+1, BUT_PT1.y+1), Scalar(128, 128, 128), 1);
        }
        else
        {
            putText(pcaShapeOp, "Constrain", Point(pcaShapeOp.cols - 150,30 ), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(128, 128, 128));
            line(pcaShapeOp, BUT_PT1, BUT_PT2, Scalar(128, 128, 128), 2);
            line(pcaShapeOp, BUT_PT2, BUT_PT3, Scalar(128, 128, 128), 2);
            line(pcaShapeOp, BUT_PT3, BUT_PT4, Scalar(128, 128, 128), 1);
            line(pcaShapeOp, BUT_PT4, BUT_PT1, Scalar(128, 128, 128), 1);
        }

        imshow("PCA component sliders", pcaShapeOp);
        waitKey(1);
    }
}

int main()
{
    vector<vector<Point2f> > data;
    load_data_pts(string(LANDMARKS_DIR) + LANDMARKS_FILENAME , data);
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

    write_data_pts(PROCRUSTES_MEAN_DATA_PATH, meanMatVec);

    for(int i=0; i<dataMatGPA.size(); i++)
    {
        upscale_data(dataMatGPA[i]);
        dataMatGPA[i].reshape(2).copyTo(data[i]);
    }

    write_data_pts(PROCRUSTES_DATA_PATH, data);

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

    imwrite(string(OUTPUT_DATA_DIR) + "data_points.jpg", dataOp);
    imwrite(string(OUTPUT_DATA_DIR) + "mean_shape.jpg", meanOp);
    imwrite(string(OUTPUT_DATA_DIR) + "GPA_shapes.jpg", GPAOp);


//PCA
    Mat dataMatPCA = formatImagesForPCA(dataMatGPA);
    vector<Mat> meanMat_{meanMat};
    Mat meanMatPCA = formatImagesForPCA(meanMat_);


//    int nComponents = 5;
    double retainedVariance = 0.9;
    PCA pca(dataMatPCA, meanMatPCA, CV_PCA_DATA_AS_ROW, retainedVariance);
    PCA_save(pca, PCA_DATA_PATH);


    namedWindow("constrain pts");
//
//
//    vector<Mat> testData;
//    testData.push_back(data);
//    testDataPCA = formatImagesForPCA(testData);
//
//    Mat testDataP = pca.project(testDataPCA);



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




















