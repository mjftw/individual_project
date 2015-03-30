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

    this->orientation = get_major_axis(this->mask);
    this->centroid = get_centroid(this->mask);
    vector<Point2f> contour2f;
    Mat(this->contour).copyTo(contour2f);
    this->scaleMetric = get_scale_metric(contour2f);

    return;
}

cv::Mat SkydiverBlob::paramaters_image()
{
    cv::Mat dst = this->mask.clone();

    cv::circle(dst, this->centroid, 5, cv::Scalar(128), 2);
    int l = 50;// line length
    cv::line(dst, this->centroid, cv::Point(this->centroid.x + l*cos(orientation), this->centroid.y + l*sin(orientation)), cv::Scalar(128), 2);

    std::stringstream text("");

    text << "Centroid: (" << this->centroid.x << "," << this->centroid.y;
    text << "), Major axis: " << this->orientation << " degrees";
    text << ", Scale metric: " << this->scaleMetric;

    cv::putText(dst, text.str(), Point(0,dst.cols/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(128), 2);

    return dst;
}
