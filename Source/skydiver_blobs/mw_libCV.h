void find_contour_centroids(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Point>& output_array/*, bool hull=1*/)
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

void overlay_contours(cv::Mat& src, cv::Mat& dst, std::vector<std::vector<cv::Point> >& contours)
{
    cv::cvtColor(src, dst, CV_GRAY2BGR);

    std::vector<Point> centroids;
    find_contour_centroids(contours, centroids);

    cv::drawContours(dst, contours, -1, cv::Scalar(0,0,255), 2, 8, noArray(), INT_MAX, cv::Point());
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
