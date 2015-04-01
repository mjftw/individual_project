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

#define LANDMARKS_FILENAME "../data/landmarks.txt"

#define DEG2RAD 3.141592654/180
#define RAD2DEG 3.141592654/180

//Body points definitions
enum{WAIST=0,NECK=1,HEAD=2,L_HAND=3,L_ELBOW=4,L_KNEE=5, L_FOOT=6,R_FOOT=7,R_KNEE=8,R_ELBOW=9,R_HAND=10};

inline string get_point_name(int pt)
{
    switch(pt)
    {
        case WAIST:
            return "waist";
        break;
        case NECK:
            return "neck";
        break;
        case HEAD:
            return "head";
        break;
        case L_HAND:
            return "l_hand";
        break;
        case L_ELBOW:
            return "l_elbow";
        break;
        case L_KNEE:
            return "l_knee";
        break;
        case L_FOOT:
            return "l_foot";
        break;
        case R_FOOT:
            return "r_foot";
        break;
        case R_KNEE:
            return "r_knee";
        break;
        case R_ELBOW:
            return "r_elbow";
        break;
        case R_HAND:
            return "r_hand";
        break;
        default:
            return "ERROR";
        break;
    }
}
inline double to_rads(double degs)
{
    degs *= 3.141592654/180;
    return degs;
}

inline double to_degs(double rads)
{
    rads *= 180/3.141592654;
    return rads;
}

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
        for(int j=0; j<data[i].size(); j++)
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

    double theta = 0.5*atan2(2*u11_, u20_- u02_);
    if(use_degrees)
       theta = to_degs(theta);

    return theta;
}

//inline double get_scale_metric(vector<Point2f>& pts)
//{//centroid size
//
//    Point2f centroid = get_vec_centroid(pts);
//    double scaleMetric = 0;
//    for(int i=0; i<pts.size(); i++)
//        scaleMetric += sqrt((pts[i].x-centroid.x)*(pts[i].x-centroid.x) + (pts[i].y-centroid.y)*(pts[i].y-centroid.y));
//
//    return scaleMetric;
//}

//inline double get_major_axis(Mat& img, bool use_degrees = true)
//{
//    RotatedRect r = minAreaRect(img);
//    cout << r.angle << endl;
//    return r.angle;
//}

inline double get_scale_metric(Mat& pts)
{
    RotatedRect r = minAreaRect(pts);
    return r.size.height * r.size.width;
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

inline void median_filter_binary(Mat& src, Mat& dst, int filter_size = 3, int filter_shape = MORPH_RECT)
{
    Mat kernel = getStructuringElement(filter_shape, Size(filter_size, filter_size));
    filter2D(src, dst, CV_8U, kernel);
    threshold(dst, dst, 6, 255, THRESH_BINARY);
}

inline void plot_pts(Mat& img, vector<Point2f>& pts, Scalar colour)
{
    for(int i=0; i< pts.size(); i++)
        circle(img, pts[i], 3, colour, 2);
}

inline void draw_body_pts(Mat& img, vector<Point2f>& pts, Scalar colour)
{
    line(img, pts[WAIST], pts[NECK], colour, 1);
    line(img, pts[NECK], pts[HEAD], colour, 1);
    line(img, pts[L_HAND], pts[L_ELBOW], colour, 1);
    line(img, pts[R_HAND], pts[R_ELBOW], colour, 1);
    line(img, pts[L_KNEE], pts[L_FOOT], colour, 1);
    line(img, pts[R_KNEE], pts[R_FOOT], colour, 1);
    line(img, pts[NECK], pts[L_ELBOW], colour, 1);
    line(img, pts[NECK], pts[R_ELBOW], colour, 1);
    line(img, pts[WAIST], pts[L_KNEE], colour, 1);
    line(img, pts[WAIST], pts[R_KNEE], colour, 1);

    circle(img, pts[HEAD], 20, colour, 2);
    circle(img, pts[L_HAND], 5, colour, 2);
    circle(img, pts[R_HAND], 5, colour, 2);
    circle(img, pts[NECK], 2, colour, 2);
    circle(img, pts[WAIST], 2, colour, 2);
    circle(img, pts[L_KNEE], 2, colour, 2);
    circle(img, pts[R_KNEE], 2, colour, 2);
    circle(img, pts[L_ELBOW], 2, colour, 2);
    circle(img, pts[R_ELBOW], 2, colour, 2);
    circle(img, pts[L_FOOT], 2, colour, 2);
    circle(img, pts[R_FOOT], 2, colour, 2);
}

inline Mat formatImagesForPCA(const vector<Mat> &data)
{
    ///Taken from OpenCV PCA example program:
    ///opencv_source_code/samples/cpp/pca.cpp
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}

inline Mat reformatImageFromPCA(Mat& img, int channels=1, int rows=11)
{
    vector<double> colVec;
    vector<Point2f> pts;

    img.reshape(channels, rows);
    img.copyTo(colVec);

    for(int i=0; i<colVec.size()/2; i++)
        pts.push_back(Point2f(colVec[2*i], colVec[2*i +1]));

    return Mat(pts);
}

inline void PCA_save(const PCA& pca, string path)
{
    FileStorage fs(path, FileStorage::WRITE);
    fs << "mean" << pca.mean;
    fs << "eigenvectors" << pca.eigenvectors;
    fs << "eigenvalues" << pca.eigenvalues;
    fs.release();
}

inline void PCA_load(PCA& pca, string path)
{
    FileStorage fs(path, FileStorage::READ);
    fs["mean"] >> pca.mean;
    fs["eigenvectors"] >> pca.eigenvectors;
    fs["eigenvalues"] >> pca.eigenvalues;
    fs.release();
}

inline Point Point2f_to_Point(Point2f pt)
{
    return Point((int)pt.x, (int)pt.y);
}

inline Point2f Point_to_Point2f(Point pt)
{
    return Point(floor(pt.x + 0.5), (floor(pt.y + 0.5)));
}

inline vector<Point> Point2f_to_Point_vec(vector<Point2f>& pts)
{
    vector<Point> ptsOut;
    for(int i=0; i<pts.size(); i++)
        ptsOut.push_back(Point((int)pts[i].x, (int)pts[i].y));
    return ptsOut;
}

inline vector<Point2f> Point_to_Point2f_vec(vector<Point>& pts)
{
    vector<Point2f> ptsOut;
    for(int i=0; i<pts.size(); i++)
        ptsOut.push_back(Point(floor(pts[i].x + 0.5), floor(pts[i].y + 0.5)));
    return ptsOut;
}

inline void draw_angle(Mat& img, Point2f& pt, double theta, Scalar colour = Scalar(128, 128, 128), int length=50)
{
    circle(img, pt, 7, colour, 2);
    line(img, pt, Point(pt.x + length*cos(to_rads(theta)), pt.y + length*sin(to_rads(theta))), colour, 2);
}

inline Mat get_subimg(Mat& src, Point2f& center_pt, Point2f& ref_pt, int box_size)
{   /*TODO add code to only use & rotate bounding rectangle for speed*/
    Mat dst, rotMat, bounding, rotated, cropped;
    float theta = to_degs(atan2(ref_pt.y - center_pt.y, ref_pt.x - center_pt.x));
    RotatedRect rotRect(center_pt, Size(box_size, box_size), theta);

    rotMat = getRotationMatrix2D(rotRect.center, theta-90, 1.0);
    warpAffine(src, rotated, rotMat, src.size(), INTER_CUBIC);
    getRectSubPix(rotated, Size(box_size, box_size), rotRect.center, cropped);

//    resize(cropped, cropped, Size(), 2, 2, INTER_CUBIC); // upscale 2x

//// Uncomment in order to see the rotated rectangles on the frames
//    Point2f rectVerts[4];
//    rotRect.points(rectVerts);
//    for(int i=0; i<4; i++)
//        line(src, rectVerts[i], rectVerts[(i+1)%4], Scalar(255,255,255));

    return dst;
}

#endif //MW_LIBCV_H
