/*
  Copyright 2007 Computer Vision Lab,
  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
  All rights reserved.

  Author: Vincent Lepetit (http://cvlab.epfl.ch/~lepetit)

  This file is part of the ferns_demo software.

  ferns_demo is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  ferns_demo is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  ferns_demo; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA
*/
#ifndef ferns_h
#define ferns_h

#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//#include "fine_gaussian_pyramid.h"

#define HOG_TO_CTC_RATIO 50		// Out of 100

struct decision_rule{
  int type;			// 1: HOG bins, 2: CTC
  int idxs[6];
};

class ferns
{
 public:
  static const int maximum_number_of_octaves = 10;

  // ds_min and ds_max currently ignored (Always taking ds_min = ds_max = 0)
  ferns(int number_of_ferns, int number_of_tests_per_fern,
        int hog_img_size);

  ferns(ferns * source);

  ferns(char * filename);
  ferns(ifstream & f);
  ~ferns();
  bool correctly_read;

  bool save(char * filename);
  bool save(ofstream & f);

  int * drop(vector<float> &fisheye_HOG_descriptor, Mat& cropped_head_img);
  int * drop(float *fisheye_HOG_descriptor, Mat& cropped_head_img);
  bool drop(vector<float> &fisheye_HOG_descriptor, Mat& cropped_head_img, int * leaves_index);
  bool drop(float *fisheye_HOG_descriptor, Mat& cropped_head_img, int * leaves_index);
  /*// drop functions:
  // if pyramid->compute_full_resolution_images is set to true => do test on pyramid->full_images The ds in tests ARE CURRENTLY IGNORED.
  // otherwise => do test on pyramid->aztec_pyramid . The ds in tests ARE IGNORED.

  bool drop(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index);

  // Do NOT delete the returned pointer !!!
  int * drop(fine_gaussian_pyramid * pyramid, int x, int y, int level);
  */

  // private:
  void load(ifstream & f);
  void alloc(int number_of_ferns, int number_of_tests_per_fern);
  //void pick_random_tests(int dx_min, int dx_max, int dy_min, int dy_max, int ds_min, int ds_max);
  void pick_random_tests(int hog_length, int img_width, int img_height, int margin);
  //void precompute_D_array(int * D, IplImage * image);

  /*
  bool drop_full_images(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index);
  bool drop_aztec_pyramid(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index);
  */

  void get_decision_rule_array(decision_rule * rule, int * output_array);
  void put_array_in_decision_rule(decision_rule * rule, int * input_array);

  void visualize(string filename, Mat& img, vector<float>& hog_des, int gt_angle, int gt_category);
  Mat draw_hog(vector<float>& hog_des, vector<Point>& hog_points);

  int number_of_ferns, number_of_tests_per_fern, number_of_leaves_per_fern;
  int hog_image_size;

  decision_rule * fern_rules;
  //int * D_full_images, * D_aztec_pyramid[maximum_number_of_octaves];

  //void compute_max_d(void);
  int max_d;

  int * preallocated_leaves_index;
};

#endif
