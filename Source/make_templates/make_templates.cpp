#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../mw_libCV.h"

#define WHITE 255
#define BLACK 0

using namespace std;
using namespace cv;

void make_templates(vector<vector<Mat> >& imgs, vector<vector<Mat> >& imgs_mask)
{
    namedWindow("");
    vector<Mat> templates(imgs.size());

    FileStorage fs((string)TEMPLATES_PATH + TEMPLATES_IMG_LIST + ".xml", FileStorage::WRITE);
    vector<string> subimgNames;
    vector<string> templateNames;

    cout << "Building templates..." << endl;
    for(int i=0; i<11; i++)
    {
//        imgs[0][i] &= imgs_mask[0][i];
        templates[i] = imgs[0][i].clone();

        for(int j=1; j<imgs.size(); j++)
        {
            stringstream ss;
            ss << SUBIMG_NAME << "_" << get_point_name(i) << "_" << j << ".bmp";
            imwrite(TEMPLATES_PATH + ss.str(), imgs[j][i]);
            subimgNames.push_back(ss.str());

//            imgs[j][i] &= imgs_mask[j][i];
            //Cumulative moving average prevents template saturating over multiple additions
            templates[i] += (imgs[j][i] - templates[i])/(i+1);
        }

        stringstream ss;
        ss << TEMPLATES_NAME << "_" << get_point_name(i) << "_" << ".bmp";
        templateNames.push_back(ss.str());
        imwrite (TEMPLATES_PATH + ss.str(), templates[i]);
    }

    fs << "img_size" << imgs[0][0].cols;

    for(int i=0; i<subimgNames.size(); i++)
        fs << "subimg" << subimgNames[i];

    for(int i=0; i<templateNames.size(); i++)
        fs << "template" << templateNames[i];

    fs.release();
    return;
}

void get_skydiver_subimgs(vector<vector<Mat> >& subimgs, vector<Mat>& landmarksFrames,vector<vector<Point2f> >& pts, int roiSize,  int i)
{
    subimgs[i].resize(11);
    //0  Waist
    subimgs[i][WAIST] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][WAIST], pts[i][ref_pt(WAIST)], roiSize);
    //1  Neck
    subimgs[i][NECK] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][NECK], pts[i][ref_pt(NECK)], roiSize);
    //2  Head
    subimgs[i][HEAD] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][HEAD], pts[i][ref_pt(HEAD)], roiSize);
    //3  Left hand
    subimgs[i][L_HAND] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][L_HAND], pts[i][ref_pt(L_HAND)], roiSize);
    //4  Left elbow
    subimgs[i][L_ELBOW] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][L_ELBOW], pts[i][ref_pt(L_ELBOW)], roiSize);
    //5  Left knee
    subimgs[i][L_KNEE] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][L_KNEE], pts[i][ref_pt(L_KNEE)], roiSize);
    //6  Left foot
    subimgs[i][L_FOOT] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][L_FOOT], pts[i][ref_pt(L_FOOT)], roiSize);
    //7  Right foot
    subimgs[i][R_FOOT] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][R_FOOT], pts[i][ref_pt(R_FOOT)], roiSize);
    //8  Right knee
    subimgs[i][R_KNEE] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][R_KNEE], pts[i][ref_pt(R_KNEE)], roiSize);
    //9  Right elbow
    subimgs[i][R_ELBOW] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][R_ELBOW], pts[i][ref_pt(R_ELBOW)], roiSize);
    //10 Right hand
    subimgs[i][R_HAND] = get_subimg(landmarksFrames[div(i,4).quot], pts[i][R_HAND], pts[i][ref_pt(R_HAND)], roiSize);
}

int main()
{
    int roiSize = 60;
    bool useColour = true;

    vector<vector<Mat> > subimgs, subimgsMask;
    vector<Mat> templates;
    vector<vector<Point2f> > pts;
    vector<Mat> landmarksFrames, landmarksFramesMask;
    if(!load_data_pts((string)LANDMARKS_DIR + LANDMARKS_FILENAME, pts))
    {
        cout << "ERROR: Cannot open " << (string)LANDMARKS_DIR + LANDMARKS_FILENAME << "for reading" << endl;
        exit(EXIT_FAILURE);
    }

    landmarksFrames.resize(pts.size()/4);
    for(int i=0; i<pts.size()/4; i++)
    {
        stringstream ss;
        ss << (string)LANDMARKS_DIR + LANDMARKS_FRAMES_FILENAME << i << ".bmp";
        landmarksFrames[i] = imread(ss.str().c_str(), useColour? CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE);
    }

    landmarksFramesMask.resize(pts.size()/4);
    for(int i=0; i<pts.size()/4; i++)
    {
        stringstream ss;
        ss << (string)LANDMARKS_DIR + LANDMARKS_FRAMES_FILENAME << i << "_bin.bmp";
        landmarksFramesMask[i] = imread(ss.str().c_str(), useColour? CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE);
    }

    subimgs.resize(pts.size());
    subimgsMask.resize(pts.size());

    cout << "Extracting subimgs..." << endl;
    for(int i=0; i<pts.size(); i++)
    {
        if(pts[i].size() != 11)
        {
            cout << "ERROR: Landmark vector incorrect length" << endl;
            exit(EXIT_FAILURE);
        }
        else
        {
            get_skydiver_subimgs(subimgs, landmarksFrames, pts, roiSize, i);
            get_skydiver_subimgs(subimgsMask, landmarksFramesMask, pts, roiSize, i);
        }
    }

    make_templates(subimgs, subimgsMask);
    return 0;
}

