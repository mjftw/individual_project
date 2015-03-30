/**
  Created by Merlin Webster.
  Copyright (c) 2015 Merlin Webster. All rights reserved.
*/



#ifndef MW_LIBCV_H
#define MW_LIBCV_H

#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

inline void find_contour_centroids(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Point>& output_array/*, bool hull=1*/)
{
    output_array.clear();

    std::vector<std::vector<cv::Point> > covexHull(contours.size());
    std::vector<cv::Moments> mu(contours.size());

    for(int i=0; i<contours.size(); i++)
    {
        convexHull(contours.at(i), covexHull[i], false, true);
        mu[i] = moments(covexHull[i], false);
        output_array.push_back(cv::Point(mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00));
    }
    return;
}

inline void overlay_contours(cv::Mat& src, cv::Mat& dst, std::vector<std::vector<cv::Point> >& contours)
{
    cv::cvtColor(src, dst, CV_GRAY2BGR);

    std::vector<cv::Point> centroids;
    find_contour_centroids(contours, centroids);

    cv::drawContours(dst, contours, -1, cv::Scalar(0,0,255), 2, 8, cv::noArray(), INT_MAX, cv::Point());
    for(int i=0; i<centroids.size(); i++)
    {
        cv::circle(dst, centroids[i], 5, Scalar(255,0,0), -1, 8, 0);

        cv::RotatedRect bounding_rect = minAreaRect(contours[i]);
        cv::Point2f rect_corners[4];
        bounding_rect.points(rect_corners); // corner points of bounding rect
        for(int j=0; j<4; j++)
            cv::line(dst, rect_corners[j], rect_corners[(j+1)%4], cv::Scalar(0,255,0), 2, 8, 0);
    }

    return;
}


inline bool load_data_pts(string data_file_path, vector<vector<Point2f> >& data)
{
    ifstream dataFile(data_file_path.c_str(), ios::in);
    if(!dataFile.is_open())
    {
        cout << "Cannot open data file:" << data_file_path << endl;
        return false;
    }

    double x,y;

    for(int i=0; !dataFile.eof(); i++)
    {
        data.resize(i+1);
        for(int j=0; j<11; j++)
        {
            dataFile >> x >> y;
            data[i].push_back(Point2f(x,y));
        }
    }

    dataFile.close();
    return true;
}

inline bool write_data_pts(string data_file_path, vector<vector<Point2f> >& data)
{
    ofstream dataFile(data_file_path, ios::out | ios::trunc);
    if(!dataFile.is_open())
    {
        cout << "Cannot open data file:" << data_file_path << endl;
        return false;
    }

    for(int i=0; i<data.size(); i++)
    {
        for(int j=0; j<11; j++)
        {
            dataFile << data[i][j].x << ' ' << data[i][j].y;
            if(j<10)
                dataFile << ' ';
        }

        if(i < data.size() - 1)
            dataFile << '\n';
    }

    dataFile.close();

    return true;
}

inline void plot_pts(Mat& img, vector<Point2f>& pts, Scalar& colour)
{
    for(int i=0; i< pts.size(); i++)
        circle(img, pts[i], 3, colour, 2);
}

inline Point2f get_centroid(Mat& binary_img)
{
    //http://en.wikipedia.org/wiki/Image_moment
    //http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=minarearect#minarearect

    cv::Moments m = moments(binary_img, 1);
    return cv::Point2f(m.m10/m.m00, m.m01/m.m00);
}

inline double get_major_axis(Mat& binary_img, bool use_degrees = true)
{
    //http://en.wikipedia.org/wiki/Image_moment
    //http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=minarearect#minarearect

    cv::Moments m = moments(binary_img, 1);
    //construct covariance matrix
    //
    //  | u20_  u11_ |
    //  | u11_  u02_ |

    double u20_ = m.mu20/m.m00;
    double u02_ = m.mu02/m.m00;
    double u11_ = m.mu11/m.m00;

    double theta = 0.5*atan((2*u11_)/(u20_ - u02_));
    if(use_degrees)
        theta *= 180/3.141592654;
    return theta;
}

inline Point2f get_vec_centroid(vector<Point2f>& pts)
{
    Point2f centroid;
    for(int i=0; i<pts.size(); i++)
    {
        centroid.x += pts[i].x;
        centroid.y += pts[i].y;
    }
    centroid.x /= pts.size();
    centroid.y /= pts.size();

    return centroid;
}

inline double get_scale_metric(vector<Point2f>& pts)
{//centroid size

    Point2f centroid = get_vec_centroid(pts);
    double scaleMetric = 0;
    for(int i=0; i<pts.size(); i++)
        scaleMetric += sqrt((pts[i].x-centroid.x)*(pts[i].x-centroid.x) + (pts[i].y-centroid.y)*(pts[i].y-centroid.y));

    return scaleMetric;
}

#endif //MW_LIBCV_H
