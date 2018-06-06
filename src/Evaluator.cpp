/*
 * CSVExtractor.cpp
 *
 *  Created on: Apr 2, 2018
 *      Author: veerachart
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;


// **************************** //
// REMARKS FOR RESULT RECORDING //
// **************************** //
// Frame#, TrackedObj#, TrackedHum#, DetectedObj#, DetectedHum#, DetectedHead#, // (cont.)
//       , TrackedObjs, TrackedHums, DetectedObjs, DetectedHums, DetectedHeads, processTime\n
//
// Each TrackedObj/TrackedHum contains
// countHuman, x, y, w, h, angle, x, y, w, h, angle, heightRatio, headRatio, deltaAngle
//            |------ body -----||------ head -----| |----- head-body relationship ----|
// Each Detected* contains
// x, y, w, h, angle
//
// READING
// Read line & the first 6 numbers to know TrackedObj#, TrackedHum#, DetectedObj#, DetectedHum#, DetectedHead#
// Then use them to indicate how many data need to be read.

void ProcessEntity(struct dirent* entity, vector<string>& file_list);
void ProcessDirectory(string directory, vector<string>& file_list);

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

class Evaluator {
protected:
	ifstream &f_result, &f_gt;
	ofstream &f_write;
	size_t trackedObjectsLength, detectedObjectsLength;
	Point img_center;
	float detect_ratio;
	string path_dir;
	vector<string> file_list;
	VideoWriter outputVideo;
	string save_name;
	string blank_line;

	void drawRotatedRect(Mat &draw, RotatedRect rect, Scalar color, Point2f shift) {
		Point2f vertices[4];
		rect.points(vertices);
		for (int v = 0; v < 4; v++)
			line(draw, vertices[v] + shift, vertices[(v+1)%4] + shift, color, 2);
	}
public:
	Evaluator(ifstream &_file_result, ifstream &_file_gt, ofstream &_file_write, string folderName, string fileName) : f_result(_file_result), f_gt(_file_gt), f_write(_file_write) {
		trackedObjectsLength = 17;
		detectedObjectsLength = 5;
		img_center = Point(400,330);
		detect_ratio = 0.5;

		path_dir = "/home/veerachart/Datasets/Dataset_PIROPO/" + folderName;
		ProcessDirectory(path_dir, file_list);
		sort(file_list.begin(), file_list.end());


		save_name = "output/evaluate_" + fileName + ".avi";
		outputVideo.open(save_name, CV_FOURCC('D','I','V','X'), 10, Size(1600, 660), true);
		f_write << "frame,xb_gt,yb_gt,wb_gt,hb_gt,xh_gt,yh_gt,wh_gt,hh_gt,direction_gt,"
						 "xb_result,yb_result,wb_result,hb_result,xh_result,yh_result,wh_result,hh_result,direction_result" << endl;
		blank_line = "NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN";			// For when the ground truth is not available
	}

	bool readLineResult(vector<int> &counts, vector<RotatedRect> &bodies, vector<RotatedRect> &heads, vector<RotatedRect> &detectedObjects, vector<RotatedRect> &detectedBodies,
			vector<RotatedRect> &detectedHeads, vector<float> &relationsVec, vector<int> &directionsVec, float *sum_times, int *count_times, float *max_times) {
		counts.clear();
		bodies.clear();
		heads.clear();
		detectedObjects.clear();
		detectedBodies.clear();
		detectedHeads.clear();
		relationsVec.clear();
		directionsVec.clear();

		string line_s;
		if (getline(f_result, line_s)) {
			stringstream linestream(line_s);
			string value;
			// Read frame#
			getline(linestream, value,',');
			int frame_id = atoi(value.c_str());
			cout << frame_id << endl;
			// Read the next 5 numbers of tracked & detected things
			int numbers[5];
			for (int i = 0; i < 5; i++) {
				getline(linestream, value,',');
				numbers[i] = atoi(value.c_str());
			}

			/*cout << "\t";
			for (int i = 0; i < 5; i++) {
				cout << numbers[i] << " ";
			}
			cout << endl;*/

			string temp = linestream.str();

			int countHuman;
			RotatedRect head, body, obj;
			// Check that there are enough values for the remaining texts
			int data_len = count(temp.begin(), temp.end(), ',');
			if (data_len + 1 - 6 - 1 == numbers[0]*trackedObjectsLength + numbers[1]*trackedObjectsLength +
					numbers[2]*detectedObjectsLength + numbers[3]*detectedObjectsLength + numbers[4]*detectedObjectsLength) {
				// n commas ==> n+1 values
				// -6 ==> the first 6 values read before
				// -1 ==> the last computation time
				relationsVec.resize(3*(numbers[0] + numbers[1]));
				directionsVec.resize(3*(numbers[0] + numbers[1]));
				int num = 0;
				for (; num < 2; num ++) {
					// tracked objects/humans
					for (int i = 0; i < numbers[num]; i++) {
						float* rel_pt = &relationsVec[num*numbers[0]*3 + i*3];
						int* dir_pt = &directionsVec[num*numbers[0]*3 + i*3];
						readTrackedObjects(linestream, countHuman, body, head, rel_pt, dir_pt);
						counts.push_back(countHuman);
						bodies.push_back(body);
						heads.push_back(head);
					}
				}

				// num = 2
				// detected objects
				for (int i = 0; i < numbers[num]; i++) {
					readDetectedObjects(linestream, obj);
					detectedObjects.push_back(obj);
				}
				num++;
				// num = 3
				for (int i = 0; i < numbers[num]; i++) {
					readDetectedObjects(linestream, obj);
					detectedBodies.push_back(obj);
				}
				num++;
				// num = 4
				for (int i = 0; i < numbers[num]; i++) {
					readDetectedObjects(linestream, obj);
					detectedHeads.push_back(obj);
				}

				getline(linestream,value);		// Last one without ','
				float comp_time = atof(value.c_str());
				sum_times[numbers[2]] += comp_time;
				count_times[numbers[2]]++;
				if (comp_time > max_times[numbers[2]])
				    max_times[numbers[2]] = comp_time;
			}
			else {
				cout << "\t" << data_len + 1 - 6 -1 << endl;
			}
			return true;
		}
		else
			return false;
	}

	bool readLineGT(vector<RotatedRect> &detectedBodies, vector<RotatedRect> &detectedHeads, vector<int> &directions) {
		detectedBodies.clear();
		detectedHeads.clear();
		directions.clear();
		string line_s;
		if (getline(f_gt, line_s)) {
			stringstream linestream(line_s);
			string value;
			string temp = linestream.str();
			int data_len = count(temp.begin(), temp.end(), ',');
			RotatedRect body, head;
			if (data_len + 1 == 12) {
				getline(linestream, value,',');		// fignum
				getline(linestream, value,',');		// idx
				getline(linestream, value,',');		// person_id
				if (atoi(value.c_str()) == -1)
					return true;					// person_id == -1 means there is no person in the frame

				readGTObjects(linestream, head);
				detectedHeads.push_back(head);
				readGTObjects(linestream, body);
				detectedBodies.push_back(body);

				getline(linestream, value);			// direction
				int direction = atoi(value.c_str());
				directions.push_back(direction);
			}
			return true;
		}
		else {
			return false;
		}
	}

	void readFiles() {
		// GT side
		vector<RotatedRect> bodies_GT, heads_GT;
		vector<int> directions;

		// Result side
		vector<int> counts;
		vector<RotatedRect> bodies, heads, detectedObjects, detectedBodies, detectedHeads;
		vector<float> relationsVec;
		vector<int> directionsVec;
		
		float sum_times[4];
		int count_times[4];
		float max_times[4];
		for (int s = 0; s < 4; s++) {
		    sum_times[s] = 0.;
		    count_times[s] = 0;
		    max_times[s] = 0.;
	    }

		string temp;
		getline(f_gt, temp);			// Throw away the header

		// TP, FP, FN
		int TP_body = 0, FP_body = 0, FN_body = 0;
		int TP_head = 0, FP_head = 0, bodiesFN_head = 0;
		int TP_tracked_body = 0, FP_tracked_body = 0, FN_tracked_body = 0;
		int TP_tracked_head = 0, FP_tracked_head = 0, FN_tracked_head = 0;
		int GT_true = 0;
		int abs_error_dir = 0;
		int count_maae = 0;
		int idx = 0;

		vector<int> errors;

		Point2f result_pos(20,20);
		char result_text[20];

		while (readLineResult(counts, bodies, heads, detectedObjects, detectedBodies, detectedHeads, relationsVec, directionsVec, sum_times, count_times, max_times) &&
				readLineGT(bodies_GT, heads_GT, directions)) {			// While not the end of the file yet
			if (idx > file_list.size()) {
				break;
			}
			Mat img = imread(path_dir+file_list[idx]);
			copyMakeBorder(img,img,30,30,0,0,BORDER_REPLICATE,Scalar(0,0,0));
			/*Mat drawHead;
			img.copyTo(drawHead);
			for (int i = 0; i < detectedHeads.size(); i++) {
				drawRotatedRect(drawHead, detectedHeads[i], Scalar(63,63,255), Point2f(0,0));
			}
			char figname[30];
			sprintf(figname, "output/CheckHeads/head%04d.jpg", idx);
			imwrite(figname, drawHead);*/
			Point2f shift_drawbody(0, 0);
			Point2f shift_drawhead(img.cols, 0);
			//Point2f shift_drawdir(0, img.rows);
			Mat draw = Mat::zeros(Size(img.cols*2,img.rows), CV_8UC3);
			img.copyTo(draw.colRange(0, img.cols).rowRange(0, img.rows));
			img.copyTo(draw.colRange(img.cols, 2*img.cols).rowRange(0, img.rows));
			//img.copyTo(draw.colRange(0, img.cols).rowRange(img.rows, 2*img.rows));
			GT_true += bodies_GT.size();
			// Tracked results first
			// Body
			if (!bodies_GT.size()) {			// No human in GT --> any humans mean FP
				for (int i = 0; i < bodies.size(); i++) {
					if (counts[i] >= 3) {		// Only human
						cout << "Ghost!" << endl;
						FP_tracked_body += 1;
						FP_tracked_head += 1;
						sprintf(result_text, "FP");
						putText(draw, result_text, result_pos + shift_drawbody, FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
						putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
					}
				}
				imshow("Comparison", draw);
				waitKey(1);
				f_write << idx << "," << blank_line << endl;
				outputVideo << draw;
			}
			else if (!bodies.size()) {			// No human detected --> all GTs become FN
				FN_tracked_body += bodies_GT.size();
				FN_tracked_head += heads_GT.size();
				sprintf(result_text, "FN");
				putText(draw, result_text, result_pos + shift_drawbody, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
				putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
				imshow("Comparison", draw);
				waitKey(1);
				f_write << idx << "," << blank_line << endl;
				outputVideo << draw;
			}
			else {
				vector<int> used_track;
				for (int gt = 0; gt < bodies_GT.size(); gt++) {
					RotatedRect rect_gt = bodies_GT[gt];
					drawRotatedRect(draw, rect_gt, Scalar(0,255,0), shift_drawbody);

					// Find the track with the best overlap
					int best_track = -1;
					float best_ratio = -1;
					vector<Point2f> intersect_points, hull;
					float intersectArea;
					for (int track = 0; track < bodies.size(); track++) {
						if (find(used_track.begin(), used_track.end(), track) != used_track.end())
							continue;				// already used
						if (counts[track] < 3)		// Not yet a human
							continue;
						RotatedRect rect_body = bodies[track];
						if (rotatedRectangleIntersection(rect_gt, rect_body, intersect_points) != INTERSECT_NONE) {
							// Intersect
							convexHull(intersect_points, hull);
							intersectArea = contourArea(hull);
						}
						else {
							intersectArea = 0.;
						}

						float ratio = intersectArea/(rect_gt.size.area() + rect_body.size.area() - intersectArea);
						if (ratio > best_ratio) {
							best_track = track;
							best_ratio = ratio;
						}
					}
					if (best_track == -1) {
						FN_tracked_body += 1;
						FN_tracked_head += 1;
						sprintf(result_text, "FN");
						putText(draw, result_text, result_pos + shift_drawbody, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
						putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
						imshow("Comparison", draw);
						waitKey(1);
						f_write << idx << "," << blank_line << endl;
						outputVideo << draw;
						continue;
					}

					if (best_ratio > detect_ratio) {
						TP_tracked_body += 1;
						sprintf(result_text, "TP");
						putText(draw, result_text, result_pos + shift_drawbody, FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
						used_track.push_back(best_track);
					}
					else {
						cout << best_ratio << endl;
						FN_tracked_body += 1;
						sprintf(result_text, "FN");
						putText(draw, result_text, result_pos + shift_drawbody, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
					}

					drawRotatedRect(draw, bodies[best_track], Scalar(0,0,255), shift_drawbody);

					// Use best_track to evaluate head and direction
					// Head
					if (counts[best_track] < 3) {
						FN_tracked_head += 1;
						sprintf(result_text, "FN");
						putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
						imshow("Comparison", draw);
						waitKey(1);
						f_write << idx << "," << blank_line << endl;
						outputVideo << draw;
						continue;
					}
					RotatedRect head_gt = heads_GT[gt];
					drawRotatedRect(draw, head_gt, Scalar(0,255,0), shift_drawhead);
					RotatedRect rect_head = heads[best_track];
					drawRotatedRect(draw, rect_head, Scalar(0,0,255), shift_drawhead);
					if (rotatedRectangleIntersection(head_gt, rect_head, intersect_points) != INTERSECT_NONE) {
						// Intersect
						convexHull(intersect_points, hull);
						intersectArea = contourArea(hull);
						float ratio = intersectArea/(head_gt.size.area() + rect_head.size.area() - intersectArea);
						if (ratio > detect_ratio) {
							TP_tracked_head += 1;
							sprintf(result_text, "TP");
							putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
						}
						else {
							FN_tracked_head += 1;
							FP_tracked_head += 1;
							sprintf(result_text, "FN");
							putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
						}
					}
					else {
						FN_tracked_head += 1;
						FP_tracked_head += 1;
						sprintf(result_text, "FN");
						putText(draw, result_text, result_pos + shift_drawhead, FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
					}
					// direction
					int dir_gt = directions[gt];
					int dir_result = directionsVec[best_track*3 + 2];		// The third value for tracked direction

					int angle_result = round( 180. - dir_result + (rect_head.angle < 0 ? rect_head.angle + 360. : rect_head.angle));
					arrowedLine(draw, rect_head.center + shift_drawhead, rect_head.center + shift_drawhead + 50.*Point2f(sin(angle_result*CV_PI/180.), -cos(angle_result*CV_PI/180.)), Scalar(0,0,255), 2);
					int angle_gt = round(180. - dir_gt + (head_gt.angle < 0 ? head_gt.angle + 360. : head_gt.angle));
					arrowedLine(draw, head_gt.center + shift_drawhead, head_gt.center + shift_drawhead + 50.*Point2f(sin(angle_gt*CV_PI/180.), -cos(angle_gt*CV_PI/180.)), Scalar(0,255,0), 2);
					//int diff = dir_gt - dir_result;
					int diff = angle_gt - angle_result;
					// bring it to [0,360]
					while (diff < 0)
						diff += 360;
					// select the smaller angle
					if (diff > 180)
						diff = 360 - diff;
					if (diff < 0 || diff > 180)
						cout << "Wrong!: " << diff << ", " << angle_gt << ", " << angle_result << endl;
					abs_error_dir += diff;
					errors.push_back(diff);
					count_maae++;
					sprintf(result_text, "%d", diff);
					putText(draw, result_text, result_pos + shift_drawhead + Point2f(0,20), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
					imshow("Comparison", draw);
					waitKey(1);
					outputVideo << draw;
					f_write << idx << "," << rect_gt.center.x << "," << rect_gt.center.y << "," << rect_gt.size.width << "," << rect_gt.size.height << ","
										  << head_gt.center.x << "," << head_gt.center.y << "," << head_gt.size.width << "," << head_gt.size.height << "," << angle_gt;
					f_write << "," << bodies[best_track].center.x << "," << bodies[best_track].center.y << "," << bodies[best_track].size.width << "," << bodies[best_track].size.height << ","
								   << rect_head.center.x << "," << rect_head.center.y << "," << rect_head.size.width << "," << rect_head.size.height << "," << angle_result;
					f_write << endl;
				}

				for (int track = 0; track < bodies.size(); track++) {
					if (find(used_track.begin(), used_track.end(), track) != used_track.end())
						continue;				// already used
					if (counts[track] < 3)		// Not yet a human
						continue;
					cout << "Leftover" << endl;
					FP_tracked_body += 1;
					FP_tracked_head += 1;
				}
			}
			/////////////////////////////////////////////
			idx++;
		}
		cout << "Evaluation for tracking of body:" << endl;
		cout << "\tTP " << TP_tracked_body << endl;
		cout << "\tFN " << FN_tracked_body << endl;
		cout << "\tFP " << FP_tracked_body << endl;
		cout << "\tDetection rate " << double(TP_tracked_body)/double(GT_true) * 100. << "%" << endl;
		cout << endl;

		cout << "Evaluation for tracking of head:" << endl;
		cout << "\tTP " << TP_tracked_head << endl;
		cout << "\tFN " << FN_tracked_head << endl;
		cout << "\tFP " << FP_tracked_head << endl;
		cout << "\tDetection rate " << double(TP_tracked_head)/double(GT_true) * 100. << "%" << endl;
		cout << endl;

		cout << "Evaluation of direction estimation:" << endl;
		cout << "\tMAAE " << double(abs_error_dir)/double(count_maae) << " deg" << endl;
		cout << endl;

		cout << "For checking" << endl;
		cout << "\tTotal ground truth: " << GT_true << endl;
		cout << "\tUsed for calculating MAAE: " << count_maae << endl;
		cout << endl;
		
		cout << "Computation times:" << endl;
		for (int i = 0; i < 4; i++) {
		    cout << "\tWith " << i << " objects: " << count_times[i] << " frames, avg " << sum_times[i]/float(count_times[i]) << " ms." << endl;
		    cout << "\t\tmax " << max_times[i] << " ms." << endl;
	    }
	}

	void readTrackedObjects(stringstream &linestream, int &countHuman, RotatedRect &body, RotatedRect &head, float *relations, int *directions) {
		string value;
		getline(linestream, value,',');
		countHuman = atoi(value.c_str());
		float rect_temp[detectedObjectsLength];
		for (int j = 0; j < detectedObjectsLength; j++) {
			getline(linestream, value,',');
			rect_temp[j] = atof(value.c_str());
		}
		body = RotatedRect(Point2f(rect_temp[0], rect_temp[1]), Size2f(rect_temp[2], rect_temp[3]), rect_temp[4]);
		for (int j = 0; j < detectedObjectsLength; j++) {
			getline(linestream, value,',');
			rect_temp[j] = atof(value.c_str());
		}
		head = RotatedRect(Point2f(rect_temp[0], rect_temp[1]), Size2f(rect_temp[2], rect_temp[3]), rect_temp[4]);
		for (int j = 0; j < 3; j++) {
			getline(linestream, value,',');
			relations[j] = atof(value.c_str());
		}
		for (int j = 0; j < 3; j++) {
			getline(linestream, value,',');
			directions[j] = atoi(value.c_str());
		}
	}

	void readDetectedObjects(stringstream &linestream, RotatedRect &obj) {
		string value;
		float rect_temp[detectedObjectsLength];
		for (int j = 0; j < detectedObjectsLength; j++) {
			getline(linestream, value,',');
			rect_temp[j] = atof(value.c_str());
		}
		obj = RotatedRect(Point2f(rect_temp[0], rect_temp[1]), Size2f(rect_temp[2], rect_temp[3]), rect_temp[4]);
	}

	void readGTObjects(stringstream &linestream, RotatedRect &obj) {
		string value;
		float rect_temp[4];
		for (int j = 0; j < 4; j++) {
			getline(linestream, value,',');
			rect_temp[j] = atof(value.c_str());
		}
		obj = RotatedRect(Point2f(rect_temp[0], rect_temp[1]+30.), Size2f(rect_temp[2], rect_temp[3]), atan2(rect_temp[0]-img_center.x, img_center.y-rect_temp[1]-30.)*180./CV_PI);
	}
};

class EvaluateNoGT : protected Evaluator {
public:
    EvaluateNoGT(ifstream &_file_result, ifstream &_file_gt, ofstream &_file_write, string folderName, string fileName) : Evaluator(_file_result, _file_gt, _file_write, folderName, fileName){
    }
    bool readLineGT(vector<Point2f> &headCenters) {
        headCenters.clear();
        string line_s;
        if (getline(f_gt, line_s)) {
            stringstream linestream(line_s);
            string value;
            string temp = linestream.str();
            int data_len = count(temp.begin(), temp.end(), ',');
            Point2f head;
            int x, y;
            getline(linestream, value, ',');        // fignum
            while(true) {
                getline(linestream, value, ',');    // x
                x = atoi(value.c_str());
                getline(linestream, value, ',');    // y
                y = atoi(value.c_str());
                if (x == 0 && y == 0)               // no data
                    break;
                head = Point2f(x + 30.,y);
                headCenters.push_back(head);
            }
            return true;
        }
        else {
            return false;
        }
    }

    void readFiles() {
        // GT side
        vector<RotatedRect> bodies_GT, heads_GT;
        vector<int> directions;

        // Result side
        vector<int> counts;
        vector<RotatedRect> bodies, heads, detectedObjects, detectedBodies, detectedHeads;
        vector<float> relationsVec;
        vector<int> directionsVec;
        vector<Point2f> headGT;

        float sum_times[4];
        int count_times[4];
        float max_times[4];
        for (int s = 0; s < 4; s++) {
            sum_times[s] = 0.;
            count_times[s] = 0;
            max_times[s] = 0.;
        }

        int idx = 0;

        vector<int> errors;
        //for(int r = 0; r < 273-189; r++) {
        for(int r = 0; r < 189-180; r++) {
            //readLineGT(headGT);
            //readLineResult(counts, bodies, heads, detectedObjects, detectedBodies, detectedHeads, relationsVec, directionsVec, sum_times, count_times, max_times);
            f_write << idx << "," << blank_line << endl;
            idx++;
        }

        while (readLineResult(counts, bodies, heads, detectedObjects, detectedBodies, detectedHeads, relationsVec, directionsVec, sum_times, count_times, max_times) &&
            readLineGT(headGT)) {          // While not the end of the file yet
            if (idx > file_list.size()) {
                break;
            }

            if (!headGT.size()) {            // No GT for head center
                f_write << idx << "," << blank_line << endl;
            }
            else if (!bodies.size()) {       // No body from the result
                f_write << idx << "," << blank_line << endl;
            }
            else {
                vector<int> used_track;
                for (int gt = 0; gt < headGT.size(); gt++) {
                    Point2f centerGT = headGT[gt];

                    int best_track = -1;
                    float best_dist = 10000;

                    for (int track = 0; track < heads.size(); track++) {
                        if (find(used_track.begin(), used_track.end(), track) != used_track.end())
                            continue;               // already used
                        if (counts[track] < 3)      // Not yet a human
                            continue;
                        float dist = norm(heads[track].center - centerGT);
                        if (dist < best_dist) {
                            best_track = track;
                            best_dist = dist;
                        }
                    }
                    if (best_track != -1) {
                        used_track.push_back(best_track);
                        RotatedRect rect_head = heads[best_track];
                        int dir_result = directionsVec[best_track*3 + 2];       // The third value for tracked direction
                        int angle_result = round( 180. - dir_result + (rect_head.angle < 0 ? rect_head.angle + 360. : rect_head.angle));
                        f_write << idx << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                                              << centerGT.x << "," << centerGT.y << "," << 0 << "," << 0 << "," << 0;
                        f_write << "," << bodies[best_track].center.x << "," << bodies[best_track].center.y << "," << bodies[best_track].size.width << "," << bodies[best_track].size.height << ","
                                       << rect_head.center.x << "," << rect_head.center.y << "," << rect_head.size.width << "," << rect_head.size.height << "," << angle_result;
                        f_write << endl;
                    }
                    else {
                        f_write << idx << "," << blank_line << endl;
                    }
                }
            }
            idx++;
        }
    }
};

int main (int argc, char ** argv) {
	//ifstream file_result("output/Results/omni1A_test1_FEHOG_fixbugcompute.csv");
    int camNum, testNum;
    int month, date;
    bool useFEHOG;
    string remarkStr;
    if (argc < 6 || argc > 7) {
        cerr << "USAGE: ./build/Evaluator cam_num[1-3] test_num[1-12] month[1-12] date[1-31] FEHOG[f/o] {remark}" << endl;
        return -1;
    }
    int temp = atoi(argv[1]);
    if (temp > 0 && temp < 4)
        camNum = temp;
    else {
        cerr << "Wrong camera number, the first argument should be 1, 2, or 3." << endl;
        return -1;
    }
    temp = atoi(argv[2]);
    if (temp > 0 && temp < 13)
        testNum = temp;
    else {
        cerr << "Wrong test number, the second argument should be between 1 and 12, inclusive." << endl;
        return -1;
    }
    temp = atoi(argv[3]);
    if (temp > 0 && temp < 13)
        month = temp;
    else {
        cerr << "Wrong month, the third argument should be between 1 and 12, inclusive." << endl;
        return -1;
    }
    temp = atoi(argv[4]);
    if (temp > 0 && temp < 32)
        date = temp;
    else {
        cerr << "Wrong date, the fourth argument should be between 1 and 31, inclusive." << endl;
        return -1;
    }
    ostringstream buffer;
    buffer << "omni_" << camNum << "A/omni" << camNum <<"A_test" << testNum << "/";
    if (strlen(argv[5]) == 1) {
        if (argv[5][0] == 'f') {
            useFEHOG = true;
        }
        else if (argv[5][0] == 'o') {
            useFEHOG = false;
        }
        else {
            useFEHOG = true;
            cout << "use \'f\' for FEHOG or \'o\' for original HOG. Now use FEHOG as default." << endl;
        }
    }
    if (argc == 7) {
        remarkStr = string(argv[6]);
    }
    else
        remarkStr = "regular";

    char mmdd[5];
    sprintf(mmdd, "%02d%02d", month, date);
    char fileName[100];
    sprintf(fileName, "omni%dA_test%d_%s_%s%s", camNum, testNum, (useFEHOG ? "FEHOG" : "originalHOG"), mmdd, remarkStr.c_str());
    char inFileName[100];
    sprintf(inFileName, "output/Results/%s.csv", fileName);
    char outFileName[100];
    sprintf(outFileName, "output/Results/evaluate_%s.csv", fileName);
    cout << fileName << endl;
    cout << inFileName << endl;
    cout << outFileName << endl;
    cout << buffer.str() << endl;

    char gtFileName[100];
    sprintf(gtFileName, "/home/veerachart/Datasets/Dataset_PIROPO/omni_%dA/Ground_Truth_Annotations/groundTruth_omni%dA_test%d.csv", camNum, camNum, testNum);
    cout << gtFileName << endl;

	ifstream file_result(inFileName);
	ifstream file_gt("/home/veerachart/Datasets/PIROPO_annotated/omni_1A/omni1A_test1/with_directions/direction_label_full.csv");
//ifstream file_gt(gtFileName);
	ofstream file_xysizedir(outFileName);
	Evaluator extractor(file_result, file_gt, file_xysizedir, buffer.str(), string(fileName));
	//EvaluateNoGT extractor(file_result, file_gt, file_xysizedir, buffer.str(), string(fileName));

	extractor.readFiles();

	return 0;
}


