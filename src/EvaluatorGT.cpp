/*
 * EvaluateGT.cpp
 *
 *  Created on: Apr 10, 2018
 *      Author: veerachart
 */


#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "ferns.h"
#include "fern_based_classifier.h"

using namespace std;
using namespace cv;

void ProcessEntity(struct dirent* entity, vector<string>& file_list);
void ProcessDirectory(string directory, vector<string>& file_list);
Mat visualize(Mat orig_img, Mat cropped_enlarged, int result_angle, int result_cat, double dir_angle, int orig_angle, int orig_cat, vector<float>descriptors, vector<float> descriptors_original);
Mat draw_hog(vector<float>& hog_des);

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


class EvaluatorGT {
private:
	fern_based_classifier * classifier;
	ifstream &f_gt;
	ofstream &f_write;
	Point img_center;
	string path_dir, path_cropped;
	vector<string> file_list;
	VideoWriter outputVideo;
	string save_name;
	int hog_size;
	FisheyeHOGDescriptor hog;
	HOGDescriptor hog_original;
	string blank_line;

	void drawRotatedRect(Mat &draw, RotatedRect rect, Scalar color, Point2f shift) {
		Point2f vertices[4];
		rect.points(vertices);
		for (int v = 0; v < 4; v++)
			line(draw, vertices[v] + shift, vertices[(v+1)%4] + shift, color, 2);
	}
public:
	EvaluatorGT(ifstream &_file_gt, ofstream &_file_write) : f_gt(_file_gt), f_write(_file_write) {
		char classifier_name[] = "classifiers/classifier_acc_400-4";
		classifier = new fern_based_classifier(classifier_name);
		hog_size = classifier->hog_image_size;
		hog = FisheyeHOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
		hog_original = HOGDescriptor(Size(hog_size,hog_size), Size(hog_size/2,hog_size/2), Size(hog_size/4,hog_size/4), Size(hog_size/4,hog_size/4), 9);
		img_center = Point(400,330);

		path_dir = "/home/veerachart/Datasets/Dataset_PIROPO/omni_1A/omni1A_test1/";
		path_cropped = "/home/veerachart/Datasets/PIROPO_annotated/omni_1A/omni1A_test1/with_directions/head/";
		ProcessDirectory(path_dir, file_list);
		sort(file_list.begin(), file_list.end());

		save_name = "output/omni1A_test1_evaluate_GTonly_full.avi";
		outputVideo.open(save_name, CV_FOURCC('D','I','V','X'), 10, Size(1600, 660), true);
		f_write << "frame,xb_gt,yb_gt,wb_gt,hb_gt,xh_gt,yh_gt,wh_gt,hh_gt,direction_gt,"
						 "xb_result,yb_result,wb_result,hb_result,xh_result,yh_result,wh_result,hh_result,direction_result" << endl;
		blank_line = "NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN";			// For when the ground truth is not available
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
		vector<float> computationTimes;

		string temp;
		getline(f_gt, temp);			// Throw away the header

		// TP, FP, FN
		int GT_true = 0;
		int abs_error_dir = 0, abs_error_dir_fehog = 0;
		int count_maae = 0;
		int idx = 0;

		vector<int> errors, errors_fehog;

		Point2f result_pos(20,20);
		char result_text[20];

		while (readLineGT(bodies_GT, heads_GT, directions)) {			// While not the end of the file yet
			if (idx > file_list.size()) {
				break;
			}
			Mat img = imread(path_dir+file_list[idx]);
			copyMakeBorder(img,img,30,30,0,0,BORDER_REPLICATE,Scalar(0,0,0));
			char crop_name[20];
			Point2f shift_drawhog(0, 0);
			Point2f shift_drawfehog(img.cols, 0);
			Mat draw = Mat::zeros(Size(img.cols*2,img.rows), CV_8UC3);
			img.copyTo(draw.colRange(0, img.cols).rowRange(0, img.rows));
			img.copyTo(draw.colRange(img.cols, 2*img.cols).rowRange(0, img.rows));
			GT_true += bodies_GT.size();
			// Tracked results first
			// Body
			vector<int> used_track;
			if (bodies_GT.size() == 0) {
				imshow("Comparison", draw);
				waitKey(1);
				outputVideo << draw;
				f_write << idx << "," << blank_line << endl;
			}
			else {
				for (int gt = 0; gt < bodies_GT.size(); gt++) {
					RotatedRect rect_gt = bodies_GT[gt];

					RotatedRect head_gt = heads_GT[gt];
					drawRotatedRect(draw, head_gt, Scalar(0,255,0), shift_drawhog);
					drawRotatedRect(draw, head_gt, Scalar(0,255,0), shift_drawfehog);
					Mat visual;
					img.copyTo(visual);
					drawRotatedRect(visual, head_gt, Scalar(255,0,0), shift_drawhog);

					// direction from cropped GT (Original HOG used);
					Mat rotated, crop;
					Mat M = getRotationMatrix2D(head_gt.center, head_gt.angle, 1.0);
					warpAffine(img, rotated, M, img.size());
					getRectSubPix(rotated, head_gt.size, head_gt.center, crop);
					resize(crop, crop, Size(hog_size,hog_size));
					/*sprintf(crop_name, "014%03d_%02dh.jpg", idx, gt);
					crop = imread(path_cropped+crop_name);
					if (crop.empty())
						continue;
					resize(crop, crop, Size(hog_size,hog_size));*/
					vector<float> descriptors_original;
					hog_original.compute(crop, descriptors_original);
					int output_class_original, output_angle_original;
					classifier->recognize_interpolate(descriptors_original, crop, output_class_original, output_angle_original);

					int dir_gt = directions[gt];
					int dir_result = output_angle_original;

					int angle_gt = round(180. - dir_gt + (head_gt.angle < 0 ? head_gt.angle + 360. : head_gt.angle));
					int angle_result = round(180. - dir_result + (head_gt.angle < 0 ? head_gt.angle + 360. : head_gt.angle));

					int diff = dir_gt - dir_result;
					// bring it to [0,360]
					while (diff < 0)
						diff += 360;
					// select the smaller angle
					if (diff > 180)
						diff = 360 - diff;
					if (diff < 0 || diff > 180)
						cout << "Wrong!: " << diff << ", " << dir_gt << ", " << dir_result << endl;
					abs_error_dir += diff;
					errors.push_back(diff);
					sprintf(result_text, "%d", diff);
					putText(draw, result_text, result_pos + shift_drawhog, FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
					arrowedLine(draw, head_gt.center + shift_drawhog, head_gt.center + shift_drawhog + 50.*Point2f(sin(angle_gt*CV_PI/180.), -cos(angle_gt*CV_PI/180.)), Scalar(0,255,0), 2);
					arrowedLine(draw, head_gt.center + shift_drawhog, head_gt.center + shift_drawhog + 50.*Point2f(sin(angle_result*CV_PI/180.), -cos(angle_result*CV_PI/180.)), Scalar(0,0,255), 2);


					// Direction from GT (FEHOG used);
					vector<RotatedRect> ROIs;
					vector<float> descriptors;
					ROIs.push_back(head_gt);
					hog.compute(img, descriptors, ROIs);
					int output_class, output_angle;
					classifier->recognize_interpolate(descriptors, crop, output_class, output_angle);

					int dir_fehog = output_angle;
					int angle_fehog = round(180. - dir_fehog + (head_gt.angle < 0 ? head_gt.angle + 360. : head_gt.angle));

					int diff_fehog = dir_gt - dir_fehog;
					// bring it to [0,360]
					while (diff_fehog < 0)
						diff_fehog += 360;
					// select the smaller angle
					if (diff_fehog > 180)
						diff_fehog = 360 - diff_fehog;
					if (diff_fehog < 0 || diff_fehog > 180)
						cout << "Wrong!: " << diff_fehog << ", " << dir_gt << ", " << dir_fehog << endl;
					abs_error_dir_fehog += diff_fehog;
					errors_fehog.push_back(diff_fehog);
					count_maae++;
					sprintf(result_text, "%d", diff_fehog);
					putText(draw, result_text, result_pos + shift_drawfehog, FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
					arrowedLine(draw, head_gt.center + shift_drawfehog, head_gt.center + shift_drawfehog + 50.*Point2f(sin(angle_gt*CV_PI/180.), -cos(angle_gt*CV_PI/180.)), Scalar(0,255,0), 2);
					arrowedLine(draw, head_gt.center + shift_drawfehog, head_gt.center + shift_drawfehog + 50.*Point2f(sin(angle_fehog*CV_PI/180.), -cos(angle_fehog*CV_PI/180.)), Scalar(0,0,255), 2);

					//Mat view = visualize(visual, crop, output_angle, output_class, head_gt.angle, output_angle_original, output_class_original, descriptors, descriptors_original);
					//mshow("Visualization", view);

					imshow("Comparison", draw);
					waitKey(1);
					outputVideo << draw;
					f_write << idx << "," << rect_gt.center.x << "," << rect_gt.center.y << "," << rect_gt.size.width << "," << rect_gt.size.height << ","
										  << head_gt.center.x << "," << head_gt.center.y << "," << head_gt.size.width << "," << head_gt.size.height << "," << angle_gt;
					f_write << "," << rect_gt.center.x << "," << rect_gt.center.y << "," << rect_gt.size.width << "," << rect_gt.size.height << ","
								   << head_gt.center.x << "," << head_gt.center.y << "," << head_gt.size.width << "," << head_gt.size.height << "," << angle_result;//angle_fehog;
					f_write << endl;
				}
			}
			idx++;
		}


		cout << "Evaluation of direction estimation from GT bounding boxes:" << endl;
		cout << "\tUsing Fisheye HOG" << endl;
		cout << "\t\tMAAE " << double(abs_error_dir_fehog)/double(count_maae) << " deg" << endl;
		cout << "\tUsing Original HOG" << endl;
		cout << "\t\tMAAE " << double(abs_error_dir)/double(count_maae) << " deg" << endl;
		cout << endl;

		cout << "For checking" << endl;
		cout << "\tTotal ground truth: " << GT_true << endl;
		cout << "\tUsed for calculating MAAE: " << count_maae << endl;
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

int main (int argc, char ** argv) {
	ifstream file_gt("/home/veerachart/Datasets/PIROPO_annotated/omni_1A/omni1A_test1/with_directions/direction_label_full.csv");
	ofstream file_xysizedir("output/Results/GTHOG_full.csv");
	EvaluatorGT extractor(file_gt, file_xysizedir);

	extractor.readFiles();

	return 0;
}

Mat visualize(Mat orig_img, Mat cropped_image, int result_angle, int result_cat, double dir_angle, int orig_angle, int orig_cat, vector<float> descriptors, vector<float> descriptors_original) {
	Mat base_img(900,1600,CV_8UC3,Scalar(64,64,64));
	Mat cropped_enlarged;
	resize(cropped_image, cropped_enlarged, Size(200,200));
	Point img_origin(50,50);
	Point cropped_origin(1100,50);
	Point direction_center(1100,500);
	Point direction_center_original(1400,500);
	int length = 100;

	orig_img.copyTo(base_img(Rect(img_origin, orig_img.size())));
	cropped_enlarged.copyTo(base_img(Rect(cropped_origin, cropped_enlarged.size())));
	char size_txt[10];
	sprintf(size_txt, "%d", cropped_image.rows);
	putText(base_img, size_txt,Point(1050,160), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(size_txt, "%d", cropped_image.cols);
	putText(base_img, size_txt,Point(1180,30), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	char angle_txt[50];
	sprintf(angle_txt, "Angle: %3d", result_angle);
	putText(base_img, angle_txt, Point(950,300), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(angle_txt, "Category: %d", result_cat);
	putText(base_img, angle_txt, Point(950,330), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	char angle_txt_original[50];
	sprintf(angle_txt_original, "Angle: %3d", orig_angle);
	putText(base_img, angle_txt_original, Point(1250,300), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);
	sprintf(angle_txt_original, "Category: %d", orig_cat);
	putText(base_img, angle_txt_original, Point(1250,330), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 1);

	Point direction_end = direction_center+Point(length*cos((dir_angle-90.)*CV_PI/180.), length*sin((dir_angle-90.)*CV_PI/180.));
	Point direction_end_original = direction_center_original+Point(length*cos((dir_angle-90.)*CV_PI/180.), length*sin((dir_angle-90.)*CV_PI/180.));

	line(base_img, direction_center, direction_end, Scalar(255,255,255), 2);
	arrowedLine(base_img, direction_end, direction_end+Point(50*cos((dir_angle+90.-result_angle)*CV_PI/180.),50*sin((dir_angle+90.-result_angle)*CV_PI/180.)), Scalar(0,0,255),2);

	line(base_img, direction_center_original, direction_end_original, Scalar(255,255,255), 2);
	arrowedLine(base_img, direction_end_original, direction_end_original+Point(50*cos((dir_angle+90.-orig_angle)*CV_PI/180.),50*sin((dir_angle+90.-orig_angle)*CV_PI/180.)), Scalar(0,0,255),2);

	Mat hog_draw = draw_hog(descriptors);
	Mat hog_draw_original = draw_hog(descriptors_original);
	hog_draw.copyTo(base_img(Rect(Point(950,700),hog_draw.size())));
	hog_draw_original.copyTo(base_img(Rect(Point(1250,700),hog_draw.size())));

	return base_img;
	//imshow(filename, base_img);
	//waitKey(0);
}

Mat draw_hog(vector<float>& hog_des) {
	Mat hog_mat(120,216,CV_8UC3);
	/*cout << "[";
	for (int i = 0; i < 323; i++)
		cout << hog_des[i] << ",";
	cout << hog_des[323] << "]" << endl;*/
	int index = 0;
	for (int col = 0; col < 3; col++) {
		for (int row = 0; row < 3; row++) {
			for (int cell_col = 0; cell_col < 2; cell_col++) {
				for (int cell_row = 0; cell_row < 2; cell_row++) {
					for (int bin = 0; bin < 9; bin++) {
						//cout << index << " ";
						Point top_left((col*2*9 + cell_col*9 + bin)*4, (row*2*1 + cell_row)*20);
						uchar pixel_value = hog_des[index]*512;
						//unsigned int pixel_value = index%256;
						//cout << pixel_value << ",";
						Mat pad(20,4,CV_8UC3,Scalar(pixel_value, pixel_value, pixel_value));
						pad.copyTo(hog_mat(Rect(top_left, Size(pad.cols,pad.rows))));
						index++;
					}
				}
			}
		}
	}
	//imshow("HOG", hog_mat);
	//waitKey(0);
	return hog_mat;
}
