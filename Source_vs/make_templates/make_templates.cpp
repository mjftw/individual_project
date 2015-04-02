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

vector<Mat> make_templates(vector<vector<Mat> >& imgs)
{
    vector<Mat> templates(imgs.size());

    FileStorage fs((string)TEMPLATES_PATH + TEMPLATES_IMG_LIST + ".xml", FileStorage::WRITE);
    vector<string> subimgNames;
    vector<string> templateNames;

    for(int i=0; i<11; i++)
    {
       templates[i] = imgs[0][i].clone();
        for(int j=1; j<imgs.size(); j++)
        {
            stringstream ss;
            ss << (string)SUBIMG_NAME << "_" << get_point_name(i) << "_" << j << ".bmp";
            imwrite(TEMPLATES_PATH + ss.str(), imgs[j][i]);
            subimgNames.push_back(ss.str());

           add(templates[i], imgs[j][i], templates[i]);
        }

        templates[i] /= imgs.size();

//        imshow("", templates[i]);
//        waitKey(0);
        stringstream ss;
        ss << (string)TEMPLATES_NAME << "_" << get_point_name(i) << "_" << ".bmp";
        templateNames.push_back(ss.str());
        imwrite (TEMPLATES_PATH + ss.str(), templates[i]);
    }

    fs << "img_size" << imgs[0][0].cols;

    for(int i=0; i<subimgNames.size(); i++)
        fs << "subimg" << subimgNames[i];

    for(int i=0; i<templateNames.size(); i++)
        fs << "template" << templateNames[i];

    fs.release();
    return templates;
}

int main()
{
    namedWindow("");

    int roiSize = 70;

    vector<vector<Mat> > subimgs;
    vector<Mat> templates;
    vector<vector<Point2f> > pts;
    vector<Mat> landmarksFrames;
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
        landmarksFrames[i] = imread(ss.str().c_str());
    }

    subimgs.resize(pts.size());

    for(int i=0; i<pts.size(); i++)
    {
        if(pts[i].size() != 11)
        {
            cout << "ERROR: Landmark vector incorrect length" << endl;
            exit(EXIT_FAILURE);
        }
        else
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
    }
//    for(int i=0; i<subimgs.size(); i++)
//    {
//        imshow("", subimgs[i][R_FOOT]);
//        waitKey(0);
//    }

    templates = make_templates(subimgs);


}

