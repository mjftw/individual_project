#ifndef SKYDIVERBLOB_H
#define SKYDIVERBLOB_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "../mw_libCV.h"

class SkydiverBlob
{
    public:
        SkydiverBlob();
        SkydiverBlob(std::vector<cv::Point> Contour, int Src_rows, int Src_cols);

        void approx_parameters(std::vector<cv::Point> Contour, int Src_rows, int Src_cols);
        cv::Mat paramaters_image();

        std::vector<cv::Point> contour;
        cv::Mat mask;
        cv::Point centroid;
        double orientation;
        double scaleMetric;
        cv::Point translation; //translation offset
};

#endif // SKYDIVERBLOB_H
