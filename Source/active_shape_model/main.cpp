#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "libs.h"

using namespace std;
using namespace cv;

vector<vector<Point> > load_data(string data_file_path)
{
    ifstream dataFile(data_file_path.c_str(), ios::in);
    if(!dataFile.is_open())
    {
        cout << "Cannot open data file:" << data_file_path << endl;
        exit(EXIT_FAILURE);
    }

    int x,y;
    vector<vector<Point> > data;


    for(int i=0; !dataFile.eof(); i++)
    {
        data.resize(i+1);
        for(int j=0; j<11; j++)
        {
            dataFile >> x >> y;
            data[i].push_back(Point(x,y));
        }
    }

    dataFile.close();

    return data;
}

void translatePt(Point& pt, int trans_x, int trans_y)
{
    pt.x += trans_x;
    pt.y += trans_y;

    return;
}

void scalePt(Point& pt, Point& origin, float amount)
{

}

void rotatePt(Point& pt, Point& origin, float angle)
{

}

int main()
{
    vector<vector<Point> > data = load_data("data_points.txt");

    for(int i=0; i<data.size(); i++)
        for(int j=0; j<11; j++)
        {

        }





    return 0;
}





















