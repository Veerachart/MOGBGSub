#include <opencv2/imgproc/imgproc.hpp>     //make sure to include the relevant headerfiles
//#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cvaux.h>
#include <math.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
//#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <time.h>
#include <dirent.h>
#include <iostream>
#include <ctype.h>

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

using namespace cv;
//using namespace Eigen;
using namespace std;

bool compareContourAreas ( vector<Point> contour1, vector<Point> contour2 ) {
    double i = contourArea(Mat(contour1));
    double j = contourArea(Mat(contour2));
    return ( i > j );
}

enum HogMode {
    HEAD = 0,
    BODY = 1
};

class BGSub {
    Mat fgMaskMOG2;
    BackgroundSubtractorMOG2 pMOG2;
    
    Mat img_gray;
    //Mat img_hsv;
    Mat temp;
    Mat img_thresholded_b, img_thresholded;
    
    Mat show;
    Mat contour_show;
    float scale;
    
    double u0, v0;
    
    double area_threshold;
    
    //std::ofstream *logfile;
    double t_zero;
    
    double f1,f2,f3;
    
    Point2f img_center;
    
    /*// Blue
    int iLowH_1;
    int iHighH_1;

    int iLowS_1;
    int iHighS_1;

    int iLowV_1;
    int iHighV_1;
    
    // Skin
    int iLowH_skin;
    int iHighH_skin;

    int iLowS_skin;
    int iHighS_skin;

    int iLowV_skin;
    int iHighV_skin;*/
    
    int dilation_size;
    
    VideoWriter outputVideo;
    bool save_video;
    
    FisheyeHOGDescriptor hog;
    bool toDraw;
    int mode;
    
    public:
        BGSub(bool _toDraw, int _mode){
            //ROS_INFO("Tracker created.");
            area_threshold = 30;
            
            /*time_t now = time(0);
            struct tm* timeinfo;
            timeinfo = localtime(&now);
            char buffer[80];
            char videoName[80];
            if (camera == "left") {
                strftime(buffer,80,"/home/otalab/logdata/%Y%m%d-%H%M_left.csv", timeinfo);
                strftime(videoName,80,"/home/otalab/logdata/%Y%m%d-%H%M_left.avi", timeinfo);
            }
            else if (camera == "right") {
                strftime(buffer,80,"/home/otalab/logdata/%Y%m%d-%H%M_right.csv", timeinfo);
                strftime(videoName,80,"/home/otalab/logdata/%Y%m%d-%H%M_right.avi", timeinfo);
            }*/
            //logfile = new std::ofstream(buffer);
            //*logfile << "time,f1,f2,f3,ftot\n";
            
            // HSV color detect
            // Control window for adjusting threshold values
            /*
            iLowH_1 = 100;
            iHighH_1 = 110;

            iLowS_1 = 100;
            iHighS_1 = 255;

            iLowV_1 = 60;
            iHighV_1 = 240;
            
            iLowH_skin = 2;
            iHighH_skin = 20;

            iLowS_skin = 40;
            iHighS_skin = 230;

            iLowV_skin = 30;
            iHighV_skin = 200;
            
            namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
            
            cvCreateTrackbar("LowH1", "Control", &iLowH_1, 179); //Hue (0 - 179)
            cvCreateTrackbar("HighH1", "Control", &iHighH_1, 179);

            cvCreateTrackbar("LowS1", "Control", &iLowS_1, 255); //Saturation (0 - 255)
            cvCreateTrackbar("HighS1", "Control", &iHighS_1, 255);

            cvCreateTrackbar("LowV1", "Control", &iLowV_1, 255); //Value (0 - 255)
            cvCreateTrackbar("HighV1", "Control", &iHighV_1, 255);
            
            if (save_video) {
                outputVideo.open(videoName, CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
                if (!outputVideo.isOpened()) {
                    ROS_ERROR("Could not write video.");
                    return;
                }
            }*/
            pMOG2 = BackgroundSubtractorMOG2(1000, 16, true);
            pMOG2.set("backgroundRatio", 0.8);
            toDraw = _toDraw;
            mode = _mode;
            
            if (mode == BODY)
                hog.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
            else if (mode == HEAD)
                hog.load("/home/veerachart/HOG_Classifiers/head_fastHOG.yaml");
            else {
                cout << "Given wrong mode. Used BODY" << endl;
                mode = BODY;
                hog.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
            }
        }
            
        void processImage (Mat &input_img) {
            if (img_center == Point2f() )
                img_center = Point2f(input_img.cols/2, input_img.rows/2);
            Mat original_img;
            input_img.copyTo(original_img);
            //cvtColor(cv_ptr->image, img_hsv, CV_BGR2HSV);
            //cvtColor(input_img, img_gray, CV_BGR2GRAY);
            
            //////////////// BLIMP DETECT PART ////////////////
            /*inRange(img_hsv, Scalar(iLowH_1, iLowS_1, iLowV_1), Scalar(iHighH_1, iHighS_1, iHighV_1), img_thresholded_b); //Threshold the image, Blue
            morphologyEx(img_thresholded_b, img_thresholded_b, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
            morphologyEx(img_thresholded_b, img_thresholded_b, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
            
            Mat frame;
            cv_ptr->image.copyTo(frame, img_thresholded_b);
            imshow("Frame", frame);
            
            vector<vector<Point> > contours_blimp;
            vector<Vec4i> hierarchy;
            findContours(img_thresholded_b, contours_blimp, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));
            
            //vector<Point2f> centers;
            Point2f blimp_center(0,0);
            double area_max = 0;
            int blimp_contour_idx = -1;
            
            for( int i = 0; i< contours_blimp.size(); i++ )
            {
                double area = contourArea(contours_blimp[i]);
                Moments mu = moments(contours_blimp[i], true);
                if (area < 100)
                    continue;           // Too small
                if (area > area_max){
                    blimp_center.x = mu.m10/mu.m00;
                    blimp_center.y = mu.m01/mu.m00;
                    area_max = area;
                    blimp_contour_idx = i;
                }
                circle(cv_ptr->image, Point(mu.m10/mu.m00, mu.m01/mu.m00), 4, Scalar(0,255,0));
            }
            
            if (area_max > 0)
            {
                circle(cv_ptr->image, blimp_center, 6, Scalar(0,0,255));
                
                point_msg.header.stamp = ros::Time::now();
                point_msg.point.x = blimp_center.x;
                point_msg.point.y = blimp_center.y;
                center_pub_.publish(point_msg);
            }*/
            ////////////////////////////////////////////////
            
            
            //pMOG2(img_gray, fgMaskMOG2);
            pMOG2(input_img, fgMaskMOG2);
            threshold(fgMaskMOG2, fgMaskMOG2, 128, 255, THRESH_BINARY);
                        
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, Mat::ones(5,5,CV_8U), Point(-1,-1), 2);
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
            
            vector<vector<Point> > contours_foreground;
            findContours(fgMaskMOG2.clone(), contours_foreground, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            if(contours_foreground.size() > 0){
                std::sort(contours_foreground.begin(), contours_foreground.end(), compareContourAreas);
                
                vector<RotatedRect> humans;
                vector<double> weights;
                vector<float> descriptors;
                
                vector<RotatedRect> objects, rawBoxes;
                double threshold = 0.5;
                groupContours(contours_foreground, objects, rawBoxes, threshold);
            
                if (mode == BODY)
                    hog.detectAreaMultiScale(input_img, objects, humans, weights, descriptors, Size(20,40), Size(100,200), -0.2);
                else if (mode == HEAD)
                    hog.detectAreaMultiScale(input_img, objects, humans, weights, descriptors, Size(16,16), Size(50,50), 8.33);

                if (toDraw) {
                    for(int i = 0; i < contours_foreground.size(); i++){
                        drawContours(input_img, contours_foreground, i, Scalar(0,255,0), 1, CV_AA);
                    }
                    for (int i = 0; i < humans.size(); i++) {
                        Point2f rect_points[4];
                        humans[i].points(rect_points);
                        for(int j = 0; j < 4; j++)
                            line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),2,8);
                    }
                    for (int i = 0; i < objects.size(); i++) {
                        Point2f rect_points[4];
                        objects[i].points(rect_points);
                        for(int j = 0; j < 4; j++)
                            line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);
                        rawBoxes[i].points(rect_points);
                        for(int j = 0; j < 4; j++)
                            line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0),1,8);
                /*for(int i = 0; i < contours_foreground.size(); i++){
                    double area = contourArea(contours_foreground[i]);
                    if(area < area_threshold){        // Too small contour
                        continue;
                    }
                    RotatedRect rect;
                    rect = minAreaRect(contours_foreground[i]);
                    Point2f rect_points[4];
                    rect.points(rect_points);
                    for(int j = 0; j < 4; j++)
                        line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);*/
                    
                    //char text[50];
                    
                    /*
                    // Intersection between blimp's contour and human's contour
                    // To avoid including blimp as human
                    Mat intersect = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
                    Mat blimp_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
                    Mat foreground_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
                    if (blimp_contour_idx >= 0)
                        drawContours(blimp_mask, contours_blimp, blimp_contour_idx, Scalar(255), CV_FILLED);
                    drawContours(foreground_mask, contours_foreground, i, Scalar(255), CV_FILLED);
                    intersect = blimp_mask & foreground_mask;
                    vector<vector<Point> > contours_intersect;
                    findContours(intersect.clone(), contours_intersect, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                    double intersect_area = 0;
                    if (contours_intersect.size()) {
                        for (int j = 0; j < contours_intersect.size(); j++) {
                            intersect_area += contourArea(contours_intersect[j]);
                        }
                    }
                    //morphologyEx(foreground_mask, foreground_mask, MORPH_DILATE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
                    
                    if (intersect_area < 0.4*area) {
                        // The overlap of the foreground blob is less than 40% of the blimp (now arbitrary # TODO get better number
                        //Moments m = moments(contours_foreground[i]);
                        if (fabs(rect.center.x - u0) + fabs(rect.center.y - v0) < 20) {
                            // Around the center, the orientation of the ellipse can be in any direction, depending on the direction the person is looking to
                            // TODO
                            point.x = rect.center.x;
                            point.y = rect.center.y;
                            point.z = 0.f;
                            //point.z = (float)membershipValue;
                            detected_points.points.push_back(point);
                        }
                        else {
                            double angle, diff_angle, azimuth_angle, height, width;
                            azimuth_angle = atan((rect.center.y-v0)/(rect.center.x-u0))*180.0/PI;
                            
                            if(rect.size.width < rect.size.height) {
                                //angle = acos(fabs(((rect.center.x-u0)*cos((rect.angle-90.0)*PI/180.0) + (rect.center.y-v0)*sin((rect.angle-90.0)*PI/180.0))/sqrt(std::pow(rect.center.x-u0,2) + std::pow(rect.center.y-v0,2)))) * 180.0/PI;
                                angle = rect.angle;
                                if (angle < 0.0)
                                    angle += 90.0;
                                else
                                    angle -= 90.0;
                                height = rect.size.height;
                                width = rect.size.width;
                            }
                            else {
                                //angle = acos(fabs(((rect.center.x-u0)*cos(rect.angle*PI/180.0) + (rect.center.y-v0)*sin(rect.angle*PI/180.0))/sqrt(std::pow(rect.center.x-u0,2) + std::pow(rect.center.y-v0,2)))) * 180.0/PI;
                                angle = rect.angle;
                                height = rect.size.width;
                                width = rect.size.height;
                            }
                            diff_angle = angle - azimuth_angle;
                            if (diff_angle > 150.0)
                                diff_angle -= 180.0;
                            else if (diff_angle < -150.0)
                                diff_angle += 180.0;
                            
                            // Writing on image for debug
                            sprintf(text, "%.2lf", diff_angle);
                            putText(cv_ptr->image, text, rect.center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0),2);
                            sprintf(text, "%.2lf %.2lf", rect.angle, rect.size.width/rect.size.height);
                            putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255),2);
                            sprintf(text, "%.2lf", atan((rect.center.y-v0)/(rect.center.x-u0))*180.0/PI);
                            putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(127,255,0),2);
                            //
                                
                            if (fabs(diff_angle) < 30.0) {
                                // orientation less than 15 degree from the radial direction -- supposed to be human
                                Point2f head_center = Point(rect.center.x + 3.*height/8.*sgn(rect.center.x-u0)*cos(fabs(angle)*PI/180.), rect.center.y + 3.*height/8.*sgn(rect.center.y-v0)*sin(fabs(angle)*PI/180.));
                                RotatedRect ROI(head_center, Size(height/4., 3.*width/4.), angle);
                                //Point2f rect_points[4];
                                ROI.points(rect_points);
                                Point points[4];
                                for(int j = 0; j < 4; j++) {
                                    line( cv_ptr->image, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),2,8);
                                    points[j] = rect_points[j];
                                }
                                Mat temp_mask = Mat::zeros(img_gray.rows, img_gray.cols, CV_8U);
                                Rect ROI_rect = ROI.boundingRect();
                                Rect head_matrix_bound(Point(std::max(0,ROI_rect.x), max(0,ROI_rect.y)), Point(std::min(img_gray.cols, ROI_rect.x+ROI_rect.width), std::min(img_gray.cols, ROI_rect.y+ROI_rect.height)));
                                //rectangle(temp_mask, head_matrix_bound, Scalar(255), -1);
                                fillConvexPoly(temp_mask, points, 4, Scalar(255));
                                rectangle(cv_ptr->image, head_matrix_bound, Scalar(255), 1);
                                temp_mask = temp_mask & foreground_mask;
                                float head_area = sum(temp_mask)[0]/255.0;
                                Mat temp_head(original_img, head_matrix_bound);
                                Mat temp_head_hsv;
                                img_hsv.copyTo(temp_head_hsv, temp_mask);
                                Mat head_hsv(temp_head_hsv, head_matrix_bound);
                                //img_hsv.copyTo(temp_head_hsv, temp_mask);
                                Mat face_mat;//, hair_mat;
                                drawContours(cv_ptr->image, contours_foreground, i, Scalar(0,255,0), 2, CV_AA);        // Draw in green
                                inRange(head_hsv, Scalar(iLowH_skin, iLowS_skin, iLowV_skin), Scalar(iHighH_skin, iHighS_skin, iHighV_skin), face_mat); //Threshold the image, skin
                                //inRange(head_hsv, Scalar(0, 2, 2), Scalar(180, 180, 80), hair_mat); //Threshold the image, hair
                                morphologyEx(face_mat, face_mat, MORPH_CLOSE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
                                //morphologyEx(hair_mat, hair_mat, MORPH_CLOSE, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
                                
                                //Point2f face_center;//, hair_center;
                                bool face_found = false;
                                double face_area;//, hair_area;
                                
                                vector<vector<Point> > contours;
                                findContours(face_mat.clone(), contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                                if (contours.size() > 0) {
                                    std::sort(contours.begin(), contours.end(), compareContourAreas);
                                    //Moments mu = moments(contours[0], true);
                                    //face_center = Point(mu.m10/mu.m00, mu.m01/mu.m00);
                                    //circle(cv_ptr->image, face_center+Point2f(ROI.boundingRect().x,ROI.boundingRect().y), 4, Scalar(255,255,255));
                                    
                                    face_area = contourArea(contours[0]);
                                    //sprintf(text, "%.4f, %.4f", face_area, head_area);
                                    //putText(cv_ptr->image, text, head_center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255) ,2);
                                    Mat face_show;
                                    temp_head.copyTo(face_show, face_mat);
                                    imshow("Face", face_show);
                                    if (face_area >= 0.4*head_area) {
                                        // Face is large enough -- half of the head
                                        face_found = true;
                                    }
                                }
                            }
                            else {
                                drawContours(cv_ptr->image, contours_foreground, i, Scalar(0,0,255), 1, CV_AA);        // Draw in red
                                sprintf(text, "%.2lf", diff_angle);
                                putText(cv_ptr->image, text, rect.center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0),2);
                                sprintf(text, "%.2lf %.2lf", rect.angle, rect.size.width/rect.size.height);
                                putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,255),2);
                                sprintf(text, "%.2lf", azimuth_angle);
                                putText(cv_ptr->image, text, Point(rect.center.x, rect.center.y+60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(127,255,0),2);
                            }
                        }
                        
                    }
                    else {
                        // Supposed to be blimp. Draw for debug
                        drawContours(cv_ptr->image, contours_foreground, i, Scalar(255,0,0), 2, CV_AA);        // Draw in blue
                    }*/
                    }
                }
            }
            if (toDraw) {
                imshow("FG Mask MOG 2", fgMaskMOG2);
                imshow("Detection", input_img);
            }
            //waitKey(1);
            //std::cout << end-begin << std::endl;
        }
        
        void groupContours ( vector< vector<Point> > inputContours, vector<RotatedRect> &outputBoundingBoxes, vector<RotatedRect> &rawBoundingBoxes, double distanceThreshold=1.0 ) {
            if (!inputContours.size())
                return;
            // inputContours should be sorted in descending area order
            outputBoundingBoxes.clear();
            rawBoundingBoxes.clear();
            //int j;
            for (vector< vector<Point> >::iterator it = inputContours.begin(); it != inputContours.end(); ++it) {
                if (contourArea(*it) < area_threshold)          // Too small to be the seed
                    break;
                vector<Point> contour_i = *it;
                RotatedRect rect_i = minAreaRect(contour_i);
                Point2f center_i = rect_i.center;
                double r_i = max(rect_i.size.width, rect_i.size.height) /2.;
                vector< vector<Point> >::iterator it_j = it+1;
                while (it_j != inputContours.end()) {
                    vector<Point> contour_j = *it_j;
                    RotatedRect rect_j = minAreaRect(contour_j);
                    Point2f center_j = rect_j.center;
                    double r_j = max(rect_j.size.width, rect_j.size.height) /2.;
                    double d_ij = norm(center_i - center_j);        // Distance between 2 contours
                    if ((d_ij - r_i - r_j)/(r_i+r_j) < distanceThreshold) {
                        // Close - should be combined
                        //cout << "\tMerged: " << it-inputContours.begin() << " and " << it_j-inputContours.begin() << endl;
                        contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
                        inputContours.erase(it_j);
                    }
                    else {
                        ++it_j;
                    }
                }

                //if (contourArea(contour_i) < area_threshold) {
                //    continue;
                //}
                    
                RotatedRect rect = minAreaRect(contour_i);
                Point2f center = rect.center;
                float w = rect.size.width;
                float h = rect.size.height;
                float phi = rect.angle;
                rawBoundingBoxes.push_back(rect);
                float theta = atan2(center.x - img_center.x, img_center.y - center.y) *180./CV_PI;
                float delta = abs(phi - theta) * CV_PI/180.;
                if (delta > CV_PI/2.) {            // width < height --> 90 deg change
                    float temp = w;
                    w = h;
                    h = temp;
                    delta -= CV_PI/2.;
                }
                float w_aligned = h*sin(delta) + w*cos(delta);
                w_aligned *= 1.5;
                float h_aligned = h*cos(delta) + w*sin(delta);
                h_aligned *= 1.5;
                outputBoundingBoxes.push_back(RotatedRect(center, Size(w_aligned, h_aligned), theta));
            }
        }
};

void ProcessDirectory(string directory, vector<string>& file_list);
void ProcessEntity(struct dirent* entity, vector<string>& file_list);

int main (int argc, char **argv) {
    string path_dir;
    bool toDraw;
    int mode;
    if( argc == 4 ) {
        cout << argc << endl;
    	path_dir = argv[1];
    	if (atoi(argv[2]) == 0)
    	    toDraw = false;
	    else
	        toDraw = true;
        if (strcmp(argv[3], "head") == 0)
            mode = HEAD;
        else if (strcmp(argv[3], "body") == 0)
            mode = BODY;
        else {
            cout << "Wrong mode specified. Selected body as default." << endl;
            mode = BODY;
        }
    }
    else if ( argc == 3) {
    	path_dir = "/home/veerachart/Datasets/Dataset_PIROPO/omni_1A/omni1A_test12/";
    	if (atoi(argv[1]) == 0)
    	    toDraw = false;
	    else
	        toDraw = true;
        if (strcmp(argv[2], "head") == 0)
            mode = HEAD;
        else if (strcmp(argv[2], "body") == 0)
            mode = BODY;
        else {
            cout << "Wrong mode specified. Selected body as default." << endl;
            mode = BODY;
        }
    }
    else {
        cerr << "ERROR, wrong arguments." << endl;
        cout << "Usage: ./BGSub [path_dir] draw(0 or 1) mode(head or body)" << endl;
    }
    BGSub BG_subtractor = BGSub(toDraw, mode);
    //cout << fixed;

    vector<string> file_list;
    ProcessDirectory(path_dir, file_list);
    sort(file_list.begin(), file_list.end());
    Mat frame;
    int idx = 0;
    
    for(;;)
    {
        frame = imread(path_dir+file_list[idx]);
        //int64 start = getTickCount();
        if( frame.empty() )
            break;
            
        int64 start = getTickCount();
        BG_subtractor.processImage(frame);
        int64 time = getTickCount() - start;
        cout << double(time)/getTickFrequency() << endl;
        
        if (toDraw) {
            char c = waitKey(1);
            if (c == 27)
                break;
        }
        idx++;
    }
    return 0;
}

void ProcessDirectory(string directory, vector<string>& file_list) {
	DIR* dir = opendir(directory.c_str());

	if (NULL == dir) {
		cout << "Could not open directory: " << directory.c_str() << endl;
		return;
	}

	dirent* entity = readdir(dir);

	while (entity != NULL) {
		ProcessEntity(entity, file_list);
		entity = readdir(dir);
	}

	closedir(dir);
}

void ProcessEntity(struct dirent* entity, vector<string>& file_list) {
	if (entity->d_type == DT_DIR) {
		if (entity->d_name[0] == '.')
			return;

		ProcessDirectory(string(entity->d_name), file_list);
		return;
	}

	if (entity->d_type == DT_REG) {
		if (entity->d_name[0] == 'R')		// Not image file (README.txt)
			return;
		//ProcessFile(string(entity->d_name));
		//cout << string(entity->d_name) << ", " << endl;
		file_list.push_back(string(entity->d_name));
		return;
	}

	cout << "Not a file or a directory" << endl;
}
