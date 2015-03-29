#ifndef SKYDIVERBLOB_H
#define SKYDIVERBLOB_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

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
        float orientation;
        float scale;
        cv::Point translation; //translation offset

    private:
        void approx_orientation();
        void approx_scale();
        void approx_translation();
};

#endif // SKYDIVERBLOB_H
