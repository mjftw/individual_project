#include "skydiverblob.h"

SkydiverBlob::SkydiverBlob()
{

}

SkydiverBlob::SkydiverBlob(std::vector<cv::Point> Contour, int Src_rows, int Src_cols)
{
    approx_parameters(Contour, Src_rows, Src_cols);
}

void SkydiverBlob::approx_parameters(std::vector<cv::Point> Contour, int Src_rows, int Src_cols)
{
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat temp(Src_rows, Src_cols, CV_8UC1, cv::Scalar::all(0));
    contours.push_back(Contour);
    cv::drawContours(temp, contours, 0, cv::Scalar(255), -1);
    this->mask = temp.clone();
    this->contour = Contour;

    approx_orientation();
    approx_scale();
    approx_translation();

    return;
}

void SkydiverBlob::approx_orientation()
{
    //http://en.wikipedia.org/wiki/Image_moment
    //http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=minarearect#minarearect

    cv::Moments m = moments(this->mask, 1);

    //construct covariance matrix
    //
    //  | u20_  u11_ |
    //  | u11_  u02_ |

    double u20_ = m.mu20/m.m00;
    double u02_ = m.mu02/m.m00;
    double u11_ = m.mu11/m.m00;
    this->orientation = 0.5*atan((2*u11_)/(u20_ - u02_)) * (180/3.141592654);

    //also get centroid for free
    this->centroid = cv::Point(m.m10/m.m00, m.m01/m.m00);

}

void SkydiverBlob::approx_scale()
{

}

void SkydiverBlob::approx_translation()
{

}

cv::Mat SkydiverBlob::paramaters_image()
{
    cv::Mat dst = this->mask.clone();

    cv::circle(dst, this->centroid, 5, cv::Scalar(128), 2);
    int l = 50;// line length
    cv::line(dst, this->centroid, cv::Point(this->centroid.x + l*cos(orientation), this->centroid.y + l*sin(orientation)), cv::Scalar(128), 2);


    return dst;
}
