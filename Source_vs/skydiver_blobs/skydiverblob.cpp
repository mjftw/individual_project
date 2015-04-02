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

    Mat contourMat(Contour);
    this->centroid = get_centroid(this->mask);
    this->scaleMetric = get_scale_metric(contourMat);
    this->orientation = get_major_axis(contourMat);
    check_orientation();

    return;
}

void SkydiverBlob::check_orientation()
{
    int upLength;
    int downLength;
    int n = 4; //number of points to move point each time
    int l = 0;
    Point pt;

    for(upLength=0, l=0, pt = this->centroid; this->mask.at<char>(pt) != 0;pt = Point(this->centroid.x + l*cos(to_rads(this->orientation)),
                                                                                          this->centroid.y + l*sin(to_rads(this->orientation))))
        l += n;
    upLength = l;

    for(l=0, pt = this->centroid; this->mask.at<char>(pt) != 0; l += n, pt = Point(this->centroid.x - l*cos(to_rads(this->orientation)),
                                                                                          this->centroid.y - l*sin(to_rads(this->orientation))))
        l += n;
    downLength = l;

    if(upLength < downLength)
        this->orientation += ((this->orientation > 180)? -180:180);

}

cv::Mat SkydiverBlob::paramaters_image()
{
    cv::Mat dst = this->mask.clone();

    draw_angle(dst, this->centroid, this->orientation);


    std::stringstream text("");

    text << "Centroid: (" << this->centroid.x << "," << this->centroid.y;
    text << "), Major axis: " << this->orientation << " degrees";
    text << ", Scale metric: " << this->scaleMetric;

    cv::putText(dst, text.str(), Point(0,dst.cols/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(128), 2);

    return dst;
}
