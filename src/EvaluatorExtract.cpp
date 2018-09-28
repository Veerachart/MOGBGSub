/*
 * EvaluatorExtract.cpp
 *
 *  Created on: Jul 20, 2018
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
        if (entity->d_name[0] == 'R')       // Not image file (README.txt)
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

class EvaluatorExtract {
protected:
    ifstream f_result;
    ofstream f_write;
    size_t trackedObjectsLength, detectedObjectsLength;
    Point img_center;
    string blank_line;

    void drawRotatedRect(Mat &draw, RotatedRect rect, Scalar color, Point2f shift) {
        Point2f vertices[4];
        rect.points(vertices);
        for (int v = 0; v < 4; v++)
            line(draw, vertices[v] + shift, vertices[(v+1)%4] + shift, color, 2);
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
public:
    EvaluatorExtract(string inputFileName, string outputFileName) : f_result(inputFileName.c_str(),std::ios::in), f_write(outputFileName.c_str(),std::ios::out) {
        trackedObjectsLength = 17;
        detectedObjectsLength = 5;
        img_center = Point(384,384);

        f_write << "frame,number,"          // tracked human number
                         "xb_result,yb_result,wb_result,hb_result,xh_result,yh_result,wh_result,hh_result,direction_result" << endl;
        blank_line = "NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN";         // For when the ground truth is not available
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

                getline(linestream,value);      // Last one without ','
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

    void readFiles() {
        vector<int> counts;
        vector<RotatedRect> bodies, heads, detectedObjects, detectedBodies, detectedHeads;
        vector<float> relationsVec;
        vector<int> directionsVec;
        vector<Point2f> headGT;

        float sum_times[5];
        int count_times[5];
        float max_times[5];
        for (int s = 0; s < 5; s++) {
            sum_times[s] = 0.;
            count_times[s] = 0;
            max_times[s] = 0.;
        }

        int idx = 0;

        vector<int> errors;

        while (readLineResult(counts, bodies, heads, detectedObjects, detectedBodies, detectedHeads, relationsVec, directionsVec, sum_times, count_times, max_times)) {          // While not the end of the file yet
            if (!bodies.size()) {       // No body from the result
                f_write << idx << "," << blank_line << endl;
            }
            else {
                for (int track = 0; track < heads.size(); track++) {
                    RotatedRect rect_head = heads[track];
                    int dir_result = directionsVec[track*3 + 2];       // The third value for tracked direction
                    int angle_result = round( 180. - dir_result + (rect_head.angle < 0 ? rect_head.angle + 360. : rect_head.angle));
                    while (angle_result >= 360)
                        angle_result -= 360;
                    while (angle_result < 0)
                        angle_result += 360;
                    f_write << idx << "," << track << ",";
                    f_write << bodies[track].center.x << "," << bodies[track].center.y << "," << bodies[track].size.width << "," << bodies[track].size.height << ","
                            << rect_head.center.x << "," << rect_head.center.y << "," << rect_head.size.width << "," << rect_head.size.height << "," << angle_result;
                    f_write << endl;
                }
            }
            idx++;
        }

        cout << "Computation times:" << endl;
        for (int i = 0; i < 5; i++) {
            cout << "\tWith " << i << " objects: " << count_times[i] << " frames, avg " << sum_times[i]/float(count_times[i]) << " ms." << endl;
            cout << "\t\tmax " << max_times[i] << " ms." << endl;
        }
    }
};


int main (int argc, char ** argv) {
    string inputName = "/home/veerachart/catkin_ws/src/blimp_navigation/log/0719_2146-rerun_human_tracking_stopcheck_r.csv";
    string outputName = "/home/veerachart/Desktop/output_human_tracking_r.csv";
    EvaluatorExtract extractor(inputName, outputName);

    extractor.readFiles();
}
