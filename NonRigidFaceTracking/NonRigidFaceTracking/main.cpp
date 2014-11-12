//
//  main.cpp
//  NonRigidFaceTracking
//
//  Created by Saburo Okita on 03/04/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include "Header.h"

#include "MUCTLandmark.h"
#include "PatchModels.h"
#include "ShapeModel.h"
#include "FaceDetector.h"
#include "FaceTracker.h"

using namespace std;
using namespace cv;


vector<vector<Point>> asContour( vector<Point2f>& points );
void trainFaceDetector( vector<MUCTLandmark>& landmarks );
void trainShapeModel( vector<MUCTLandmark>& landmarks );
void trainPatchModel( vector<MUCTLandmark>& landmarks );
void testFaceTracker( string video_filename );


/**
 * Before starting, please update Header.h, so that it points to proper directories
 * All the MUCT images should be placed under muct/jpg/ , e.g. muct/jpg/i000qa-fn.jpg
 * while the landmarks should be in muct/muct-landmarks/
 */
int main(int argc, const char * argv[]) {
    vector<MUCTLandmark> landmarks = MUCTLandmark::readFromCSV( path + "muct-landmarks/muct76-opencv.csv" );
    
//    trainShapeModel( landmarks );
//    trainPatchModel( landmarks );
//    trainFaceDetector( landmarks );
    testFaceTracker( "/Users/saburookita/Desktop/Untitled.mp4" );

    return 0;
}


/**
 * Return the points as a connection contour
 */
vector<vector<Point>> asContour( vector<Point2f>& points ) {
    static const vector<vector<int>> connections = {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 23, 22, 21, 0 },   /* Face outline */
        { 21, 22, 23, 24, 25, 26, 21 },                         /* Left brow */
        { 18, 17, 16, 15, 20, 19, 18 },                         /* Right brow */
        { 37, 38, 39, 40, 46, 41, 47, 42, 43, 44, 45},          /* Nose */
        { 27, 68, 28, 69, 29, 70, 30, 71, 27 },                 /* Left eye */
        { 34, 73, 33, 72, 32, 75, 35, 74, 34 },                 /* Right eye */
        { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ,48 }, /* Outer lips ridges */
        { 60, 61, 62, 63, 64, 65, 60 },                         /* Inner lips ridges */
    };
    
    vector<vector<Point>> contours;
    
    for( vector<int> section : connections ) {
        vector<Point> contour;
        for( int i = 0; i < section.size(); i++ )
            contour.push_back( points[section[i]] );
        contours.push_back( contour );
    }
    
    return contours;
}


/**
 * Train a face detector, basically applying viola jones cascade classifier
 * to find the median offset for the points.
 */
void trainFaceDetector( vector<MUCTLandmark>& landmarks ) {
    ShapeModel smodel;
    smodel.load( path + "/shape_model.yml" );
    vector<Point2f> ref_points = smodel.calcShape();
    
    Mat ref = Mat(ref_points).clone();
    ref = ref.reshape( 1, 2 * static_cast<int>(ref_points.size()) );
    
    FaceDetector detector;
    detector.train( landmarks, CASCADE_FILE, ref, 0.9, 1.1, 2, Size(30, 30), true );
    detector.save( path + "/detector.yml" );
}

/**
 * Train the Active Shape Model, and visualize it
 */
void trainShapeModel( vector<MUCTLandmark>& landmarks ) {
    ShapeModel smodel;
    smodel.train( landmarks );
    smodel.save( path + "/shape_model.yml");
    smodel.visualize();
}

/**
 * Train the patch models
 */
void trainPatchModel( vector<MUCTLandmark>& landmarks ) {
    ShapeModel smodel;
    smodel.load( path + "/shape_model.yml" );
    vector<Point2f> ref_points = smodel.calcShape();
    
    Size window_size(11, 11);
    Size patch_size(11, 11);
    
    PatchModels pmodel;
    pmodel.train( landmarks, ref_points, patch_size, window_size, false, 1.0, 1e-6, 1e-3, 1000, false );
    pmodel.save( path + "/patch_model.yml" );
    pmodel.visualize();
}

vector<vector<Point2f>> getDelanuayTriangles( vector<Point2f>& points ) {
    vector<vector<Point2f>> result;
    
    Point2f top_left    ( numeric_limits<float>::max(), numeric_limits<float>::max() );
    Point2f bottom_right( numeric_limits<float>::min(), numeric_limits<float>::min() );
    for( Point2f point : points ) {
        top_left.x      = MIN( top_left.x, point.x );
        top_left.y      = MIN( top_left.y, point.y );
        bottom_right.x  = MAX( bottom_right.x, point.x );
        bottom_right.y  = MAX( bottom_right.y, point.y );
    }
    
    Rect rect( top_left - Point2f(5, 5), bottom_right + Point2f(5, 5) );
    Subdiv2D delanuay( rect );
    delanuay.insert( points );
    
    vector<Vec6f> triangles;
    delanuay.getTriangleList( triangles );
    
    for( Vec6f triangle: triangles ) {
        Point2f a( triangle[0], triangle[1] );
        Point2f b( triangle[2], triangle[3] );
        Point2f c( triangle[4], triangle[5] );
        
        if( rect.contains( a ) && rect.contains( b ) && rect.contains( c )){
            vector<Point2f> triangle_points = { a, b, c };
            result.push_back( triangle_points );
        }
    }
    
    return result;
}

/**
 * Test the trained face tracker
 **/
void testFaceTracker( string video_filename ) {
    ShapeModel smodel;
    smodel.load( path + "/shape_model.yml" );
    
    PatchModels pmodel;
    pmodel.load( path + "/patch_model.yml" );
    
    FaceDetector detector;
    detector.load( path + "/detector.yml" );
    
    /* Face Tracker is composed of shape model (ASM), patch models, and a viola jones cascade classifier */
    FaceTracker tracker;
    tracker.shapeModel  = smodel;
    tracker.patchModels = pmodel;
    tracker.detector    = detector;
    
    vector<Size> levels = {
        Size(21, 21),
        Size(11, 11),
        Size(5, 5),
    };
    
    VideoCapture cap( video_filename );
    Mat frame;
    vector<Rect> faces;
    
    namedWindow("");
    moveWindow("", 0, 0);
    
    bool draw_connections = false;
    bool triangulation = false;
    
    while( true ){
        cap >> frame;
        
        /* Reset the video */
        if( frame.empty() ){
            cap.release();
            cap.open( video_filename );
            tracker.tracking = false;
            continue;
        }
        
        
        /* Try to track / detect face */
        if(tracker.track( frame, faces, levels, false, 20, 1e-6, 1.1, 2, Size(30, 30) )) {
            for( vector<Point2f> points: tracker.allPoints ) {
                for( Point2f point : points )
                    circle( frame, point, 1, Scalar(0, 0, 255), 1, CV_AA);
                
                /* Draw triangulation from all the points */
                if( triangulation ) {
                    vector<vector<Point2f>> triangles = getDelanuayTriangles( points );
                    for( vector<Point2f> triangle: triangles ) {
                        line( frame, triangle[0], triangle[1], Scalar(0, 255, 0) );
                        line( frame, triangle[1], triangle[2], Scalar(0, 255, 0) );
                        line( frame, triangle[2], triangle[0], Scalar(0, 255, 0) );
                    }
                }
                
                /* Draw the predefined connections between each points */
                if( draw_connections ){
                    vector<vector<Point>> contours = asContour( points );
                    for( int i = 0; i < contours.size(); i++ )
                        drawContours( frame, contours, i, Scalar(0, 255, 0));
                }
            }
            
        }
        
        imshow( "", frame );
        
        int key = waitKey(10);
        if( key == 'q' )
            break;
        else if( key == 'c' ) {
            draw_connections = !draw_connections;
            if( draw_connections )
                triangulation = false;
        }
        else if( key == 't' ){
            triangulation = !triangulation;
            if( triangulation )
                draw_connections = false;
        }
        else if( key == 'd' )
            tracker.tracking = false;
    }
}

