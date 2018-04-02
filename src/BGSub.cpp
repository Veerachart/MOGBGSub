#include <opencv2/imgproc/imgproc.hpp>     //make sure to include the relevant headerfiles
//#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <opencv2/opencv.hpp>
//#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <time.h>
#include <dirent.h>
#include <iostream>
#include <ctype.h>
#include "ferns.h"
#include "fern_based_classifier.h"

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

//////// DetectedObjects ////////
enum objectStatus {
	OBJ = 0,
	HUMAN = 1
};

class TrackedObjects {
public:
	TrackedObjects(RotatedRect objDetection, bool isHumanDetected, bool isHeadDetected, RotatedRect headDetection, Point2f imgCenter);
	Point2f PredictObject();
	Point2f UpdateObject(RotatedRect objDetection, bool isHumanDetected);			// Return predicted position of the head area
	//Point2f PredictHead(Mat &obj_vel);
	Point2f UpdateHead(RotatedRect headDetection);
	// For checking if the new detection belongs to this objects
	bool IsForThisObject(RotatedRect new_obj);
	// For checking if the new detection belongs to this objects; return distance to this object
	float distToObject(RotatedRect new_obj);
	// For checking if the new head belongs to this objects
	bool IsForThisHead(RotatedRect new_head);
	// For checking if the new head belongs to this objects; return distance to this object's head prediction
	float distToHead(RotatedRect new_head);
	// Final check at the end of the loop: if variance becomes too large, remove
	// Return true if deleted
	bool CheckAndDelete();
	float getSdBody();
	float getSdHead();
	Point2f getPointBody();
	Point2f getPointHead();
	RotatedRect getBodyROI();
	RotatedRect getHeadROI();
	Point2f getHeadVel();
	int getStatus();
	float threshold();
	int getDirection();
	void updateDirection(int estimation, int movingDirection);
	int getCount();

private:
	KalmanFilter objectKF;
	KalmanFilter headKF;
	RotatedRect objectROI;
	RotatedRect headROI;
	int status;
	int countHuman;
	float sdBody;
	float sdHead;

	// To keep body and head widths (constanted, changed only by new detections
	float bodyWidth;
	float headWidth;

	// For estimating head position & size based on human detection
	float heightRatio;		// head's height relative to window's height
	float deltaAngle;
	float headRatio;

	// For tracking of head direction
	int headDirection;

	Point2f img_center;
};

TrackedObjects::TrackedObjects(RotatedRect objDetection, bool isHumanDetected, bool isHeadDetected, RotatedRect headDetection, Point2f imgCenter) {
	objectKF = KalmanFilter(4, 2, 0);
	//headKF = KalmanFilter(3, 3, 3);
	headKF = KalmanFilter(4, 2, 0);
	sdBody = objDetection.size.width/4.;
	sdHead = objDetection.size.width/4.;

	objectKF.transitionMatrix = (Mat_<float>(4,4) << 1,0,1,0,
													 0,1,0,1,
													 0,0,1,0,
													 0,0,0,1);
	setIdentity(objectKF.measurementMatrix);
	setIdentity(objectKF.processNoiseCov, Scalar::all(16.0));
	setIdentity(objectKF.measurementNoiseCov, Scalar::all(sdBody*sdBody));
	setIdentity(objectKF.errorCovPost, Scalar::all(sdBody*sdBody));
	objectKF.statePost = (Mat_<float>(4,1) << objDetection.center.x, objDetection.center.y, 0, 0);
	bodyWidth = objDetection.size.width;
	objectROI = objDetection;

	headKF.transitionMatrix = (Mat_<float>(4,4) << 1,0,1,0,
												   0,1,0,1,
												   0,0,1,0,
												   0,0,0,1);
	setIdentity(headKF.measurementMatrix);
	setIdentity(headKF.processNoiseCov, Scalar::all(16.0));
	setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	if (isHumanDetected)
		countHuman = 1;
	else
		countHuman = 0;
	if (isHeadDetected) {
	    Point2f bodyToHead = headDetection.center - objDetection.center;
		heightRatio = norm(bodyToHead)/ objDetection.size.height;
		deltaAngle = atan2(bodyToHead.x, -bodyToHead.y) - objDetection.angle;
		headRatio = headDetection.size.width/objDetection.size.width;
		headKF.statePost = (Mat_<float>(4,1) << headDetection.center.x, headDetection.center.y, 0, 0);
		headWidth = headDetection.size.width;
		headROI = headDetection;
	}
	else {
		heightRatio = 0.3125;
		deltaAngle = 0.;
		headRatio = 0.375;
		float theta_r = (objDetection.angle + deltaAngle)*CV_PI/180.;
		Point2f headCenter = objDetection.center + heightRatio*objDetection.size.height*Point2f(sin(theta_r), -cos(theta_r));
		headKF.statePost = (Mat_<float>(4,1) << headCenter.x, headCenter.y, 0, 0);
		headWidth = headRatio*objDetection.size.width;
		headROI = RotatedRect(headCenter, Size(headWidth, headWidth), objDetection.angle);
	}
	//setIdentity(headKF.transitionMatrix);
	//setIdentity(headKF.controlMatrix);
	//setIdentity(headKF.measurementMatrix);
	//setIdentity(headKF.measurementNoiseCov, Scalar::all(sdHead*sdHead));
	//setIdentity(headKF.errorCovPost, Scalar::all(sdHead*sdHead));
	status = OBJ;

	headDirection = -1;			// Init

	img_center = imgCenter;
}

Point2f TrackedObjects::PredictObject() {
	Mat prediction = objectKF.predict();
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	objectROI = RotatedRect(Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0)),
							Size2f(bodyWidth, 2*bodyWidth),
							atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);

	Mat predictHead = headKF.predict();
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(predictHead.at<float>(0,0), predictHead.at<float>(1,0)),
						  Size2f(headWidth, headWidth),
						  atan2(predictHead.at<float>(0,0) - img_center.x, img_center.y - predictHead.at<float>(1,0)) *180./CV_PI);

	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));

	/*float r = norm(getPointBody()-img_center);
	float l = norm(getPointHead()-img_center);
	float w1 = objectROI.size.width;
	Point2f unit_r1 = (1./r) * (getPointBody()-img_center);
	Mat prediction = objectKF.predict();
	Point2f p2 = Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));
	float r2 = norm(p2-img_center);
	Point2f unit_r2 = (1./r2) * (p2-img_center);
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	objectROI = RotatedRect(p2,
							Size2f(prediction.at<float>(2,0), 2*prediction.at<float>(2,0)),
							atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);
	headKF.processNoiseCov = objectKF.errorCovPost(Rect(3,3,3,3));
	Point2f vel1(objectKF.statePost.at<float>(3,0), objectKF.statePost.at<float>(4,0));
	Point2f vel2 = vel1 + (l-r)*(unit_r2-unit_r1) - 0.4*(r2-r)*(l-r)/w1*unit_r2;

	//cout << vel1 << "\t" << vel2;
	Mat obj_vel = (Mat_<float>(3,1) << vel2.x, vel2.y, 0.4*prediction.at<float>(5,0));

	Mat predictHead = headKF.predict(obj_vel);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(predictHead.at<float>(0,0), predictHead.at<float>(1,0)),
						  Size2f(predictHead.at<float>(2,0), predictHead.at<float>(2,0)),
						  atan2(predictHead.at<float>(0,0) - img_center.x, img_center.y - predictHead.at<float>(1,0)) *180./CV_PI);
	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));*/
}

Point2f TrackedObjects::UpdateObject(RotatedRect objDetection, bool isHumanDetected) {
	Mat measurement;
	if (isHumanDetected) {
		if (status == OBJ) {
			countHuman++;
			if (countHuman >= 3)
				status = HUMAN;
		}
		setIdentity(objectKF.measurementNoiseCov, Scalar::all(objDetection.size.width*objDetection.size.width/16.));
		measurement = (Mat_<float>(2,1) << objDetection.center.x, objDetection.center.y);
		bodyWidth = objDetection.size.width;
	}
	else {
		setIdentity(objectKF.measurementNoiseCov, Scalar::all(objDetection.size.width*objDetection.size.width/16.));		// Larger variance for object
		measurement = (Mat_<float>(2,1) << objDetection.center.x, objDetection.center.y);
		bodyWidth = objectROI.size.width;
	}
	Mat corrected_state = objectKF.correct(measurement);
	objectROI = RotatedRect(Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0)),
							Size2f(bodyWidth, 2*bodyWidth),
							atan2(corrected_state.at<float>(0,0) - img_center.x, img_center.y - corrected_state.at<float>(1,0)) *180./CV_PI);

	// After updating the object, use it as weak measurement of head
	float theta_r = (objectROI.angle + deltaAngle)*CV_PI/180.;
	Point2f headCenter = objectROI.center + heightRatio*objectROI.size.height*Point2f(sin(theta_r), -cos(theta_r));
	Mat measurement_frombody = (Mat_<float>(2,1) << headCenter.x, headCenter.y);
	headWidth = headRatio*objectROI.size.width;
	setIdentity(headKF.measurementNoiseCov, Scalar::all(objectROI.size.width*objectROI.size.width/16.));
	Mat corrected_head = headKF.correct(measurement_frombody);
	headROI = RotatedRect(Point2f(corrected_head.at<float>(0,0), corrected_head.at<float>(1,0)),
						  Size(headWidth, headWidth),
						  atan2(corrected_head.at<float>(0,0) - img_center.x, img_center.y - corrected_head.at<float>(1,0)) *180./CV_PI);
	//heightRatio = 0.3125;
	//deltaAngle = 0.;
	//headRatio = 0.375;

	//Mat obj_vel = corrected_state.rowRange(3,6);
	sdBody = sqrt(min(objectKF.errorCovPost.at<float>(0,0), objectKF.errorCovPost.at<float>(1,1)));
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	return Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0));
}

/*Point2f TrackedObjects::PredictHead(Mat &obj_vel) {
	Mat prediction = headKF.predict(obj_vel);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0)),
						  Size2f(prediction.at<float>(2,0), prediction.at<float>(2,0)),
						  atan2(prediction.at<float>(0,0) - img_center.x, img_center.y - prediction.at<float>(1,0)) *180./CV_PI);
	return Point2f(prediction.at<float>(0,0), prediction.at<float>(1,0));
}*/

Point2f TrackedObjects::UpdateHead(RotatedRect headDetection) {
	Mat measurement = (Mat_<float>(2,1) << headDetection.center.x, headDetection.center.y);
	headWidth = headDetection.size.width;
	setIdentity(headKF.measurementNoiseCov, Scalar::all(headDetection.size.width*headDetection.size.width/16.));
	Mat corrected_state = headKF.correct(measurement);
	sdHead = sqrt(min(headKF.errorCovPost.at<float>(0,0), headKF.errorCovPost.at<float>(1,1)));
	headROI = RotatedRect(Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0)),
						  Size2f(headWidth, headWidth),
						  atan2(corrected_state.at<float>(0,0) - img_center.x, img_center.y - corrected_state.at<float>(1,0)) *180./CV_PI);
	heightRatio = norm(headROI.center - objectROI.center) / objectROI.size.height;
	deltaAngle = headROI.angle - objectROI.angle;
	headRatio = headROI.size.width/objectROI.size.width;
	return Point2f(corrected_state.at<float>(0,0), corrected_state.at<float>(1,0));
}

bool TrackedObjects::IsForThisObject(RotatedRect new_obj) {
	Point2f predicted_pos(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
	return (norm(new_obj.center-predicted_pos) < 3*sdBody);
}

float TrackedObjects::distToObject(RotatedRect new_obj) {
	Point2f predicted_pos(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
	return (norm(new_obj.center-predicted_pos));
}

bool TrackedObjects::IsForThisHead(RotatedRect new_head) {
	Point2f predicted_pos(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
	return (norm(new_head.center-predicted_pos) < 3*sdHead);
}

float TrackedObjects::distToHead(RotatedRect new_head) {
	Point2f predicted_pos(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
	return (norm(new_head.center-predicted_pos));
}

bool TrackedObjects::CheckAndDelete() {
	return (sdBody > 1.5*objectROI.size.width || sdHead > 1.5*objectROI.size.width); // || (status != HUMAN && sdHead > 20));			// Deviation > 30
}

float TrackedObjects::threshold() {
	float r = norm(getPointBody() - img_center);
	return max(min(136.26 - 0.4*r, 88.),12.);
}

float TrackedObjects::getSdBody() {
	return sdBody;
}

float TrackedObjects::getSdHead() {
	return sdHead;
}

Point2f TrackedObjects::getPointBody() {
	return Point2f(objectKF.statePost.at<float>(0,0), objectKF.statePost.at<float>(1,0));
}

Point2f TrackedObjects::getPointHead() {
	return Point2f(headKF.statePost.at<float>(0,0), headKF.statePost.at<float>(1,0));
}

RotatedRect TrackedObjects::getBodyROI() {
	return objectROI;
}

RotatedRect TrackedObjects::getHeadROI() {
	return headROI;
}

int TrackedObjects::getStatus() {
	return status;
}

Point2f TrackedObjects::getHeadVel() {
	return Point2f(headKF.statePost.at<float>(2,0), headKF.statePost.at<float>(3,0));
}

int TrackedObjects::getDirection() {
	return headDirection;
}

int TrackedObjects::getCount() {
	return countHuman;
}

void TrackedObjects::updateDirection(int estimation, int movingDirection) {
	if (headDirection < 0) {
		headDirection = estimation;
		return;
	}

	if (estimation < 0) {
		// When the head is out of the frame and direction cannot be estimated
		// Use only moving direction
		if (movingDirection > 0) {
			// Moving
			if (abs(movingDirection - headDirection) >= 180) {
				// crossing 0,360 line
				headDirection = cvRound((headDirection + movingDirection + 360)/2.);
				if (headDirection > 360)
					headDirection -= 360;
			}
			else {
				headDirection = cvRound((headDirection + movingDirection)/2.);
			}
		}
		return;
	}

	int diff = abs(estimation - headDirection);
	if (diff <= 45) {				// <= 45 degree change
		headDirection = cvRound((headDirection + estimation)/2.);
	}
	else if ( diff >= 315) {		// <= 45 degree change, crossing the line 0,360
		headDirection = cvRound((headDirection + estimation + 360)/2.);
		if (headDirection > 360)
			headDirection -= 360;
	}
	// More than that, update with the moving direction instead
	else {
		if (movingDirection > 0) {
			// Moving
			if (abs(movingDirection - headDirection) >= 180) {
				// crossing 0,360 line
				headDirection = cvRound((headDirection + movingDirection + 360)/2.);
				if (headDirection > 360)
					headDirection -= 360;
			}
			else {
				headDirection = cvRound((headDirection + movingDirection)/2.);
			}
		}
	}
}
/////////////////////////////////


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
    
    //double u0, v0;
    
    double area_threshold;
    
    //std::ofstream *logfile;
    //double t_zero;
    
    //double f1,f2,f3;
    
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
    
    FisheyeHOGDescriptor hog_body;
    FisheyeHOGDescriptor hog_head;
    FisheyeHOGDescriptor hog_direction;
    HOGDescriptor hog_original;
    vector<TrackedObjects> tracked_objects;
    vector<TrackedObjects> tracked_humans;
    int hog_size;
    bool toDraw;
    
    fern_based_classifier * classifier;

    public:
        BGSub(bool _toDraw, bool _toSave = false){
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

            save_video = _toSave;
            if (save_video) {
				string videoName = "omni1A_test1_newheadest.avi";
				outputVideo.open(videoName, CV_FOURCC('D','I','V','X'), 10, Size(800, 600), true);
				//outputVideo.open(videoName, CV_FOURCC('D','I','V','X'), 10, Size(768, 768), true);
				if (!outputVideo.isOpened()) {
					cerr << "Could not write video." << endl;
					return;
				}
            }
            pMOG2 = BackgroundSubtractorMOG2(1000, 50.0, true);
            pMOG2.set("backgroundRatio", 0.75);
            pMOG2.set("fTau", 0.6);
            pMOG2.set("nmixtures", 3);
            pMOG2.set("varThresholdGen", 25.0);
            pMOG2.set("fVarInit", 36.0);
            pMOG2.set("fVarMax", 5*36.0);
            toDraw = _toDraw;

            char classifier_name[] = "classifiers/classifier_acc_400-4";
			classifier = new fern_based_classifier(classifier_name);

			hog_size = classifier->hog_image_size;

            hog_body.load("/home/veerachart/HOG_Classifiers/32x64_weighted/cvHOGClassifier_32x64+hard.yaml");
            hog_head.load("/home/veerachart/HOG_Classifiers/head_fastHOG.yaml");
            hog_direction = FisheyeHOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
            hog_original = HOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
        }

        ~BGSub() {
        	delete classifier;
        }
            
        bool processImage (Mat &input_img) {
            if (img_center == Point2f() )
                img_center = Point2f(input_img.cols/2, input_img.rows/2);
            Mat original_img;
            input_img.copyTo(original_img);

            for (int track = 0; track < tracked_objects.size(); track++)
            	tracked_objects[track].PredictObject();
            for (int track = 0; track < tracked_humans.size(); track++)
				tracked_humans[track].PredictObject();

            /*Mat draw;
			input_img.copyTo(draw);

            for (int track = 0; track < tracked_objects.size(); track++) {
				Scalar color(255,255,255);
				TrackedObjects object = tracked_objects[track];
				if (object.getStatus() == HUMAN)
					color = Scalar(0,255,0);
				circle(draw, object.getPointBody(), 3, color, -1);
				Point2f rect_points[4];
				tracked_objects[track].getBodyROI().points(rect_points);
				for(int j = 0; j < 4; j++)
					line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),2,8);
				//circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

				//circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
				tracked_objects[track].getHeadROI().points(rect_points);
				for(int j = 0; j < 4; j++)
					line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(0,192,192),2,8);
				circle(draw, object.getPointHead(), 3*object.getSdHead(), Scalar(0,192,192), 2);
			}
            imshow("After predict", draw);*/
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
            //Mat save = Mat::zeros(fgMaskMOG2.size(), CV_8UC3);
			//cvtColor(fgMaskMOG2, save, CV_GRAY2BGR);
			threshold(fgMaskMOG2, fgMaskMOG2, 128, 255, THRESH_BINARY);
			//outputVideo << save;
                        
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, Mat::ones(5,5,CV_8U), Point(-1,-1), 2);
            morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, Mat::ones(3,3,CV_8U), Point(-1,-1), 1);
			imshow("FG Mask MOG 2", fgMaskMOG2);
            
            vector<vector<Point> > contours_foreground;
            findContours(fgMaskMOG2.clone(), contours_foreground, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            vector<RotatedRect> humans;
			vector<RotatedRect> heads;
			vector<double> weights;
			vector<float> descriptors;

			vector<RotatedRect> objects, rawBoxes;
			vector<RotatedRect> area_heads;					// ROI to search for heads = top half of objects

            if(contours_foreground.size() > 0){
                std::sort(contours_foreground.begin(), contours_foreground.end(), compareContourAreas);
                
                double threshold = 1.0;
                groupContours(contours_foreground, objects, rawBoxes, threshold);


                if (objects.size()) {
                	Size size_min(1000,1000), size_max(0,0);
                	for (int obj = 0; obj < objects.size(); obj++) {
                		Size temp = getHumanSize(norm(objects[obj].center - img_center));
                		if (temp.width < size_min.width)
                			size_min = temp;
                		if (temp.width > size_max.width)
                			size_max = temp;

                		float theta_r = objects[obj].angle*CV_PI/180.;
                		area_heads.push_back(RotatedRect(objects[obj].center + 0.25*objects[obj].size.height*Point2f(sin(theta_r), -cos(theta_r)), Size(objects[obj].size.width,objects[obj].size.height/2), objects[obj].angle));
                		//cout << objects[obj].center << " and " << area_heads.back().center << endl;
                	}
                	size_min -= Size(10,20);
                	size_max += Size(10,20);
                	float width_head_min = max(12., 0.375*size_min.width - 10.);
                	Size size_head_min(width_head_min, width_head_min);
                	float width_head_max = max(12., 0.375*size_max.width + 10.);
					Size size_head_max(width_head_max, width_head_max);

					//cout << size_min << " " << size_max << " " << size_head_min << " " << size_head_max << endl;

                	hog_body.detectAreaMultiScale(input_img, objects, humans, weights, descriptors, size_min, size_max, 0., Size(4,2), Size(0,0), 1.05, 1.0);

                	vector<bool> usedTrack(tracked_objects.size(),false), usedHumanTrack(tracked_humans.size(), false);
                	bool isHuman[objects.size()];
                	for (int obj = 0; obj < objects.size(); obj++)
                		isHuman[obj] = false;

                	vector<Point2f> intersect_points;
                	for (int hum = 0; hum < humans.size(); hum++) {
                		for (int obj = 0; obj < objects.size(); obj++) {
                			if ((rotatedRectangleIntersection(objects[obj], humans[hum], intersect_points)) != INTERSECT_NONE) {
                				isHuman[obj] = true;
                				break;
                			}
                		}

                		// Match detected human with tracked humans first
                		if (tracked_humans.size()) {
                			int best_track = 0;
							float best_dist = 1000;
							for (int track = 0; track < tracked_humans.size(); track++) {
								if (usedHumanTrack[track])
									continue;					// This track already got updated --> skip
								float dist = tracked_humans[track].distToObject(humans[hum]);
								if (dist < best_dist) {
									best_track = track;
									best_dist = dist;
								}
							}

							if (best_dist < 3*tracked_humans[best_track].getSdBody()) {
								// Update
								cout << "Update human with human:" << humans[hum].center << "," << humans[hum].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
								tracked_humans[best_track].UpdateObject(humans[hum], true);
								usedHumanTrack[best_track] = true;
								continue;
							}
                		}

						// Then with tracked objects
						if (tracked_objects.size()) {
							int best_track = 0;
							float best_dist = 1000;
							int best_count = 0;			// Keep the value of countHuman of the best object -- give priority to the object with more count of human
							for (int track = 0; track < tracked_objects.size(); track++) {
								if (usedTrack[track])
									continue;					// This track already got updated --> skip
								float dist = tracked_objects[track].distToObject(humans[hum]);
								int count = tracked_objects[track].getCount();
								if (dist < best_dist && count >= best_count) {
									best_track = track;
									best_dist = dist;
								}
							}

							if (best_dist < 3*tracked_objects[best_track].getSdBody()) {
								// Update
								cout << "Update object with human:" << humans[hum].center << "," << humans[hum].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
								tracked_objects[best_track].UpdateObject(humans[hum], true);
								if(tracked_objects[best_track].getStatus() == HUMAN) {
									// Converted to human after update
									// Take out from tracked_objects & add to tracked_human
									tracked_humans.push_back(tracked_objects[best_track]);
									tracked_objects.erase(tracked_objects.begin() + best_track);
									usedHumanTrack.push_back(true);		// Needed?
									usedTrack.erase(usedTrack.begin() + best_track);
									cout << "Upgrade object to human" << endl;
									continue;
								}
								else
									usedTrack[best_track] = true;
							}
							else {
								// Not within range for the existing object, create a new one
								cout << "Added new object, starting as human." << endl;
								tracked_objects.push_back(TrackedObjects(humans[hum], true, false, RotatedRect(), img_center));
								usedTrack.push_back(true);				// Needed?
							}
						}
						else {
							cout << "Added new object, starting as human." << endl;
							tracked_objects.push_back(TrackedObjects(humans[hum], true, false, RotatedRect(), img_center));
							usedTrack.push_back(true);				// Needed?
						}
                	}

                	for (int obj = 0; obj < objects.size(); obj++) {
                		if (!isHuman[obj]) {
                			// This object is not marked as a human yet, so check it as an object
                			// The same way, match with human first
                			if (tracked_humans.size()) {
								int best_track = 0;
								float best_dist = 1000;
								for (int track = 0; track < tracked_humans.size(); track++) {
									if (usedHumanTrack[track])
										continue;					// This track already got updated --> skip
									float dist = tracked_humans[track].distToObject(objects[obj]);
									if (dist < best_dist) {
										best_track = track;
										best_dist = dist;
									}
								}

								if (best_dist < 3*tracked_humans[best_track].getSdBody()) {
									// Update
									cout << "Update human with object:" << objects[obj].center << "," << objects[obj].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
									tracked_humans[best_track].UpdateObject(objects[obj], false);
									usedHumanTrack[best_track] = true;
									continue;
								}
							}

                			if (tracked_objects.size()) {
								int best_track = 0;
								float best_dist = 1000;
								int best_count = 0;
								for (int track = 0; track < tracked_objects.size(); track++) {
									if (usedTrack[track])
										continue;					// This track already got updated --> skip
									float dist = tracked_objects[track].distToObject(objects[obj]);
									int count = tracked_objects[track].getCount();
									if (dist < best_dist && count >= best_count) {
										best_track = track;
										best_dist = dist;
									}
								}

								if (best_dist < 3*tracked_objects[best_track].getSdBody()) {
									// Update
									cout << "Update object with object:" << objects[obj].center << "," << objects[obj].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
									tracked_objects[best_track].UpdateObject(objects[obj], false);
									usedTrack[best_track] = true;
								}
								else {
									// Not within range for the existing object, create a new one
									cout << "Added new object, not containing human." << endl;
									tracked_objects.push_back(TrackedObjects(objects[obj], false, false, RotatedRect(), img_center));
									usedTrack.push_back(true);		// Needed?
								}
							}
                			else {
								// New object
								cout << "Added new object, not containing human." << endl;
								tracked_objects.push_back(TrackedObjects(objects[obj], false, false, RotatedRect(), img_center));
								usedTrack.push_back(true);			// Needed?
							}
                		}
                	}

                	/*input_img.copyTo(draw);
                	for (int track = 0; track < tracked_objects.size(); track++) {
						Scalar color(255,255,255);
						TrackedObjects object = tracked_objects[track];
						if (object.getStatus() == HUMAN)
							color = Scalar(0,255,0);
						circle(draw, object.getPointBody(), 3, color, -1);
						Point2f rect_points[4];
						tracked_objects[track].getBodyROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),2,8);
						//circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

						//circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
						tracked_objects[track].getHeadROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( draw, rect_points[j], rect_points[(j+1)%4], Scalar(0,192,192),2,8);
						circle(draw, object.getPointHead(), 3*object.getSdHead(), Scalar(0,192,192), 2);
					}
					imshow("After human detect", draw);*/

                	hog_head.detectAreaMultiScale(input_img, area_heads, heads, weights, descriptors, size_head_min, size_head_max, 8.2, Size(4,2), Size(0,0), 1.05, 1.0);


                	vector<bool> humanHasHead(tracked_humans.size(), false);
                	vector<bool> objectHasHead(tracked_objects.size(), false);
                	for (int head = 0; head < heads.size(); head++) {
                		if(tracked_humans.size()) {
							int best_track = 0;
							float best_dist = 1000;
							for (int track = 0; track < tracked_humans.size(); track++) {
								if (humanHasHead[track])
									continue;					// already got head updated
								float dist = tracked_humans[track].distToHead(heads[head]);
								if (dist < best_dist) {
									best_track = track;
									best_dist = dist;
								}
							}
							if (best_dist < 3*tracked_humans[best_track].getSdHead()) {
								// Update
								cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_humans[best_track].getBodyROI().center << "," << tracked_humans[best_track].getBodyROI().size << endl;
								tracked_humans[best_track].UpdateHead(heads[head]);
								humanHasHead[best_track] = true;
								continue;
							}
							else {
								// Outside 3 SD, should we add a new object here?
								// TODO
							}
                		}
                		if(tracked_objects.size()) {
                			int best_track = 0;
							float best_dist = 1000;
							for (int track = 0; track < tracked_objects.size(); track++) {
								if (objectHasHead[track])
									continue;					// already got head updated
								float dist = tracked_objects[track].distToHead(heads[head]);
								if (dist < best_dist) {
									best_track = track;
									best_dist = dist;
								}
							}

							if (best_dist < 3*tracked_objects[best_track].getSdHead()) {
								// Update
								cout << "Update head:" << heads[head].center << "," << heads[head].size << " with " << tracked_objects[best_track].getBodyROI().center << "," << tracked_objects[best_track].getBodyROI().size << endl;
								tracked_objects[best_track].UpdateHead(heads[head]);
								objectHasHead[best_track] = true;
								continue;
							}
							else {
								// Outside 3 SD, should we add a new object here?
								// TODO
							}
						}
                	}

                	for (int track = 0; track < tracked_humans.size(); track++) {
						if (!humanHasHead[track]) {
							// If this track has not got any update for head
							// What should we do? TODO
							if (tracked_humans[track].getStatus() == HUMAN) {
								// If we are tracking this as HUMAN, head should be approximated
								// TODO
							}
						}
					}

                	for (int track = 0; track < tracked_objects.size(); track++) {
                		if (!objectHasHead[track]) {
                			// If this track has not got any update for head
                			// What should we do? TODO
                			if (tracked_objects[track].getStatus() == HUMAN) {
                				// If we are tracking this as HUMAN, head should be approximated
                				// TODO
                			}
                		}
                	}
                }
            }

            vector<float> descriptor;
            int output_class, output_angle;
            vector<int> classes, angles;
            cout << "Current objects:" << endl;
            // Final clean up for large variance objects
            for (vector<TrackedObjects>::iterator it = tracked_objects.begin(); it != tracked_objects.end(); ) {
            	cout << "\t" << it->getBodyROI().center << "," << it->getBodyROI().size << endl;
            	if (it->CheckAndDelete()) {
					it = tracked_objects.erase(it);
					cout << "An object removed" << endl;
				}
            	else
            		++it;
            }
            cout << "Current humans:" << endl;
            for (vector<TrackedObjects>::iterator it = tracked_humans.begin(); it != tracked_humans.end(); ) {
            	cout << "\t" << it->getBodyROI().center << "," << it->getBodyROI().size << endl;
            	if (it->CheckAndDelete()) {
            		it = tracked_humans.erase(it);
            		cout << "A human removed" << endl;
            	}
            	else {
            		// Calculate Ferns & direction
            		if (it->getStatus() == HUMAN) {
						Point2f head_vel = it->getHeadVel();
						RotatedRect rect = it->getHeadROI();

						float walking_dir;
						if (norm(head_vel) < 1.5) {				// TODO Threshold adjust
							walking_dir = -1.;					// Not enough speed, no clue
						}
						else {
							walking_dir = rect.angle + atan2(head_vel.x, head_vel.y)*180./CV_PI;			// Estimated walking direction relative to the radial line (0 degree head direction)
							if (walking_dir < 0)
								walking_dir += 360;					// [0, 360) range. Negative means no clue
						}
						//cout << head_vel << " Moving in " << walking_dir << " degree direction" << endl;
						//cout << rect.center << " " << rect.size << " " << rect.angle << endl;

						// Check vertices within frame
						Point2f vertices[4];
						rect.points(vertices);
						int v;
						for (v = 0; v < 4; v++) {
							if (vertices[v].x < 0 || vertices[v].x >= input_img.cols || vertices[v].y < 0 || vertices[v].y >= input_img.rows) {
								break;
							}
						}
						if (v < 4) {					// At least one out
							//cout << "OB" << endl;
							classes.push_back(-1);
							angles.push_back(-1);
							it->updateDirection(-1, int(cvRound(walking_dir)));
							++it;
							continue;
						}
						// crop head area
						Mat M = getRotationMatrix2D(rect.center, rect.angle, 1.0);
						Mat rotated, cropped;
						warpAffine(original_img, rotated, M, original_img.size(), INTER_CUBIC);
						getRectSubPix(rotated, rect.size, rect.center, cropped);
						resize(cropped, cropped, Size(hog_size,hog_size));
						/////////////////
						vector<RotatedRect> location;
						location.push_back(rect);
						hog_direction.compute(original_img, descriptor, location);
						classifier->recognize_interpolate(descriptor, cropped, output_class, output_angle, walking_dir);
						it->updateDirection(output_angle, int(cvRound(walking_dir)));
						classes.push_back(output_class);
						angles.push_back(output_angle);
            		}
            		else{
            			classes.push_back(-1);
						angles.push_back(-1);
            		}
            		++it;
            	}
            }

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
				if (toDraw) {
					//for(int i = 0; i < contours_foreground.size(); i++){
					//	drawContours(input_img, contours_foreground, i, Scalar(0,255,0), 1, CV_AA);
					//}
					for (int i = 0; i < humans.size(); i++) {
						Point2f rect_points[4];
						humans[i].points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,0),2,8);
					}
					for (int i = 0; i < heads.size(); i++) {
						Point2f rect_points[4];
						heads[i].points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,255,255),2,8);
					}
					for (int i = 0; i < objects.size(); i++) {
						Point2f rect_points[4];
						objects[i].points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255),2,8);
						rawBoxes[i].points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0),1,8);
					}

					for (int track = 0; track < tracked_objects.size(); track++) {
						Scalar color(255,255,255);
						TrackedObjects object = tracked_objects[track];
						if (object.getCount() == 0) {
							circle(input_img, object.getPointBody(), 2, color, -1);
							continue;
						}
						circle(input_img, object.getPointBody(), 2, color, -1);
						Point2f rect_points[4];
						object.getBodyROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),1,8);
						//circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

						//circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
						object.getHeadROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),1,8);
						//circle(input_img, object.getPointHead(), 3*object.getSdHead(), Scalar(255,0,0), 1);
					}

					for (int track = 0; track < tracked_humans.size(); track++) {
						Scalar color(0,255,0);
						TrackedObjects human = tracked_humans[track];
						circle(input_img, human.getPointBody(), 2, color, -1);
						Point2f rect_points[4];
						human.getBodyROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(192,192,0),1,8);
						//circle(input_img, object.getPointBody(), 3*object.getSdBody(), Scalar(192,192,0), 2);

						//circle(input_img, object.getPointHead(), 2, Scalar(192,192,192), -1);
						human.getHeadROI().points(rect_points);
						for(int j = 0; j < 4; j++)
							line( input_img, rect_points[j], rect_points[(j+1)%4], Scalar(255,255,255),1,8);
						//circle(input_img, object.getPointHead(), 3*object.getSdHead(), Scalar(255,0,0), 1);
						int dir = human.getDirection();
						float angle = (dir - human.getHeadROI().angle)*CV_PI/180.;
						arrowedLine(input_img, human.getPointHead(), human.getPointHead() + 50.*Point2f(sin(angle), cos(angle)), Scalar(255,255,255), 1);
						char buffer[10];
						sprintf(buffer, "%d, %d", angles[track], human.getDirection());
						putText(input_img, buffer , human.getBodyROI().center+50.*Point2f(-sin(human.getHeadROI().angle*CV_PI/180.),cos(human.getHeadROI().angle*CV_PI/180.)), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,255), 2);
					}

					//imshow("FG Mask MOG 2", fgMaskMOG2);
					imshow("Detection", input_img);
				}
				if(save_video)
					outputVideo << input_img;
				//waitKey(1);
				//std::cout << end-begin << std::endl;
				return (tracked_humans.size() > 0 || tracked_objects.size() > 0);
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
                    if ((d_ij - r_i - r_j) < distanceThreshold * (r_i+r_j)) {
                        // Close - should be combined
                        //cout << "\tMerged: " << it-inputContours.begin() << " and " << it_j-inputContours.begin() << endl;
                        contour_i.insert(contour_i.end(), contour_j.begin(), contour_j.end());
                        // update bounding box
                        rect_i = minAreaRect(contour_i);
                        r_i = max(rect_i.size.width, rect_i.size.height) /2.;
                        it_j = inputContours.erase(it_j);
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

                Size human_size = getHumanSize(norm(center - img_center)) + Size(10,20);
                outputBoundingBoxes.push_back(RotatedRect(center, Size(max(int(cvRound(w_aligned)),human_size.width), max(int(cvRound(h_aligned)),human_size.height)), theta));
            }
        }

        Size getHumanSize(float radius) {
        	float width;
        	if (radius > 280)
        		width = 24.;
        	else if (radius < 120)
        		width = 88.;
        	else
        		width = cvRound(136.26 - 0.4*radius);
        	return Size(width, 2*width);
        }
};

void ProcessDirectory(string directory, vector<string>& file_list);
void ProcessEntity(struct dirent* entity, vector<string>& file_list);

int main (int argc, char **argv) {
    string path_dir;
    bool toDraw;
    bool loadVideo = false;
    string video_path;
    bool continuous = false;
    if( argc == 3 ) {
    	path_dir = argv[1];
    	if (atoi(argv[2]) == 0)
    	    toDraw = false;
	    else
	        toDraw = true;
    }
    else if ( argc == 2) {
    	path_dir = "/home/veerachart/Datasets/Dataset_PIROPO/omni_1A/omni1A_test12/";
    	if (atoi(argv[1]) == 0)
    	    toDraw = false;
	    else
	        toDraw = true;
    }
    else {
    	// Video & draw
    	loadVideo = true;
    	video_path = "/home/veerachart/Videos/Test_lab.mp4";
    	toDraw = true;

        //cerr << "ERROR, wrong arguments." << endl;
        //cout << "Usage: ./BGSub [path_dir] draw(0 or 1)" << endl;
    }
    BGSub BG_subtractor = BGSub(toDraw, true);
    //cout << fixed;

    vector<string> file_list;
    if (!loadVideo) {
		ProcessDirectory(path_dir, file_list);
		sort(file_list.begin(), file_list.end());
    }
    Mat frame;
    int idx = 0;
    
    if (loadVideo) {
    	VideoCapture cap;
    	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('H','E','V','C'));
    	cap.open(video_path);
    	if (cap.isOpened()) {
    		while (cap.read(frame)) {
    			int64 start = getTickCount();
				bool toStop = BG_subtractor.processImage(frame);
				int64 time = getTickCount() - start;
				cout << double(time)/getTickFrequency() << endl;

				if (toDraw) {
					char c = waitKey((!toStop || continuous) * 1);
					if (c == 27)
						break;
					else if(c == 'c')
						continuous = !continuous;
				}
    		}
    	}
    	else {
    		cerr << "Could not open the video file" << endl;
    	}
    }
    else {
    for(;;)
    {
    	if (idx >= file_list.size())
    		break;
        frame = imread(path_dir+file_list[idx]);
        //int64 start = getTickCount();
        if( frame.data == NULL )
            break;
            
        int64 start = getTickCount();
        bool toStop = BG_subtractor.processImage(frame);
        int64 time = getTickCount() - start;
        cout << double(time)/getTickFrequency() << endl;
        
        if (toDraw) {
        	char c = waitKey((!toStop || continuous) * 1);
			if (c == 27)
				break;
			else if(c == 'c')
				continuous = !continuous;
		}
        idx++;
    }
    }
    cout << "Finished" << endl;
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
