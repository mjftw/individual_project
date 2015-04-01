#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../mw_libCV.h"

#define WHITE 255
#define BLACK 0

int main()
{
    vector<vector<Mat> > subimgs;
    vector<vector<Point2f> pts = load_data_pts(LANDMARKS_FILENAME);

    for(int i=0; i<pts.size(); i++)
    {
        if(pts[i].size() != 11)
        {
            cout << "ERROR: landmark vector incorrect length. Skipping..." << endl;
            continue;
        }

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

    }
}
