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
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <bitset>
using namespace std;
//using namespace cv;

#include "ferns.h"

// ds_min and ds_max currently ignored. Taking ds_min = ds_max = 0
ferns::ferns(int number_of_ferns, int number_of_tests_per_fern,
             int hog_img_size)
{
  alloc(number_of_ferns, number_of_tests_per_fern);
  hog_image_size = hog_img_size;
  //pick_random_tests(dx_min, dx_max, dy_min, dy_max, ds_min, ds_max);
  pick_random_tests(324,hog_img_size,hog_img_size,0);

}

ferns::ferns(ferns * source) {
	alloc(source->number_of_ferns, source->number_of_tests_per_fern);
	hog_image_size = source->hog_image_size;
	for (int k = 0; k < number_of_ferns * number_of_tests_per_fern; k++) {
		fern_rules[k].type = source->fern_rules[k].type;
		for (int i = 0; i < 6; i++) {
			fern_rules[k].idxs[i] = source->fern_rules[k].idxs[i];
		}
	}
}

ferns::ferns(char * filename)
{
  ifstream f(filename);

  if (!f.is_open()) {
    cerr << "ferns::ferns(char * filename): error reading file " << filename << "." << endl;
    correctly_read = false;
    return;
  }

  load(f);

  f.close();
}

ferns::ferns(ifstream & f)
{
  load(f);
}

ferns::~ferns() {
	delete [] preallocated_leaves_index;
	delete [] fern_rules;
}

void ferns::load(ifstream & f)
{
  int nf, nt, imsize;

  f >> nf >> nt >> imsize;
  
  cout << "> [ferns] " << nf << " ferns, " << nt << " tests per fern. " << imsize << " pixels width and height" << endl;

  alloc(nf, nt);
  hog_image_size = imsize;

  char c; do f.read(&c, 1); while (c != '.');

  int load_rule[7*number_of_ferns * number_of_tests_per_fern];

  f.read((char*)load_rule, number_of_ferns * number_of_tests_per_fern * sizeof(int) * 7);
  for (int i = 0; i < number_of_ferns * number_of_tests_per_fern; i++) {
	  put_array_in_decision_rule(&fern_rules[i], &load_rule[7*i]);
  }

  /*
  f.read((char*)DX1, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.read((char*)DY1, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.read((char*)DS1, number_of_ferns * number_of_tests_per_fern * sizeof(int));

  f.read((char*)DX2, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.read((char*)DY2, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.read((char*)DS2, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  */

  //compute_max_d();
  correctly_read = true;
}

bool ferns::save(char * filename)
{
  ofstream f(filename);
  
  if (!f.is_open()) {
    cerr << "ferns::save(char * filename): error saving file " << filename << "." << endl;

    return false;
  }

  bool result = save(f);

  f.close();

  return result;
}

bool ferns::save(ofstream & f)
{
  f << number_of_ferns << " " << number_of_tests_per_fern << " " << hog_image_size << endl;

  char dot('.'); f.write(&dot, 1);

  int save_rule[7*number_of_ferns * number_of_tests_per_fern];
  for (int i = 0; i < number_of_ferns * number_of_tests_per_fern; i++) {
	get_decision_rule_array(&fern_rules[i], &save_rule[7*i]);
  }
  for (int i = 0; i < 7*number_of_ferns * number_of_tests_per_fern; i++) {
	  cout << save_rule[i] << " ";
	  if ((i+1)%7 == 0)
		  cout << endl;
  }

  f.write((char*)save_rule, number_of_ferns * number_of_tests_per_fern * sizeof(int) * 7);

  /*f.write((char*)DX1, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.write((char*)DY1, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.write((char*)DS1, number_of_ferns * number_of_tests_per_fern * sizeof(int));

  f.write((char*)DX2, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.write((char*)DY2, number_of_ferns * number_of_tests_per_fern * sizeof(int));
  f.write((char*)DS2, number_of_ferns * number_of_tests_per_fern * sizeof(int));*/

  cout << "[ferns] Finished saving ferns..." << endl;

  return true;
}

void ferns::get_decision_rule_array(decision_rule * rule, int * output_array) {
  output_array[0] = rule->type;
  if (rule->type == 1) {
    output_array[1] = rule->idxs[0];
    output_array[2] = rule->idxs[1];
    for (int i = 3; i < 7; i++)
      output_array[i] = 0;
  }
  else {
    for (int i = 0; i < 6; i++)
	  output_array[i+1] = rule->idxs[i];
  }
  // type, idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], idxs[5]
}

void ferns::put_array_in_decision_rule(decision_rule * rule, int * input_array) {
  rule->type = input_array[0];
  if (rule->type == 1) {
    rule->idxs[0] = input_array[1];
    rule->idxs[1] = input_array[2];
    for (int i = 2; i < 6; i++)
      rule->idxs[i] = 0;
  }
  else {
    for (int i = 0; i < 6; i++)
	  rule->idxs[i] = input_array[i+1];
  }
}

/*bool ferns::drop(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index)
{
  if (pyramid->type == fine_gaussian_pyramid::full_pyramid_357)
    return drop_full_images(pyramid, x, y, level, leaves_index);
  else
    return drop_aztec_pyramid(pyramid, x, y, level, leaves_index);
}

bool ferns::drop_full_images(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index)
{
  if (pyramid->full_images[level]->width  != width_full_images ||
      pyramid->full_images[level]->height != height_full_images) {
    precompute_D_array(D_full_images, pyramid->full_images[level]);
    width_full_images  = pyramid->full_images[level]->width;
    height_full_images = pyramid->full_images[level]->height;
  }

  IplImage * smoothed_image = pyramid->full_images[level];
  int shift_x = x + pyramid->border_size;
  int shift_y = y + pyramid->border_size;

  if (shift_x < max_d || shift_y < max_d || shift_x >= smoothed_image->width - max_d  || shift_y >= smoothed_image->height - max_d)
    return false;

  unsigned char * C = (unsigned char *)(smoothed_image->imageData +
                                        shift_y * smoothed_image->widthStep +
                                        shift_x);

  for(int i = 0; i < number_of_ferns; i++) {
    int index = 0;
    int * D_ptr = D_full_images + i * 2 * number_of_tests_per_fern;
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      if (*(C + *D_ptr) < *(C + D_ptr[1])) index++;
      D_ptr += 2;
      if (j < number_of_tests_per_fern - 1) index <<= 1;
    }
    leaves_index[i] = index;
  }

  return true;
}

bool ferns::drop_aztec_pyramid(fine_gaussian_pyramid * pyramid, int x, int y, int level, int * leaves_index)
{
  int octave = level / 4; // 4 -> should not be hardcoded -> should be static const in fine_gaussian_pyramid !!!
  if (pyramid->aztec_pyramid[level]->width  != width_aztec_pyramid[octave] ||
      pyramid->aztec_pyramid[level]->height != height_aztec_pyramid[octave]) {
    precompute_D_array(D_aztec_pyramid[octave], pyramid->aztec_pyramid[level]);
    width_aztec_pyramid[octave]  = pyramid->aztec_pyramid[level]->width;
    height_aztec_pyramid[octave] = pyramid->aztec_pyramid[level]->height;
  }

  IplImage * smoothed_image = pyramid->aztec_pyramid[level];
  int shift_x = x + (pyramid->border_size >> octave);
  int shift_y = y + (pyramid->border_size >> octave);

  if (shift_x < max_d || shift_y < max_d || shift_x >= smoothed_image->width - max_d  || shift_y >= smoothed_image->height - max_d)
    return false;

  unsigned char * C = (unsigned char *)(smoothed_image->imageData +
                                        shift_y * smoothed_image->widthStep +
                                        shift_x);

  for(int i = 0; i < number_of_ferns; i++) {
    int index = 0;
    int * D_ptr = D_aztec_pyramid[octave] + i * 2 * number_of_tests_per_fern;
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      index <<= 1;
      if (*(C + *D_ptr) < *(C + D_ptr[1])) index++;
      D_ptr += 2;
    }
    leaves_index[i] = index;
  }

  return true;
}*/

int * ferns::drop(vector<float> &fisheye_HOG_descriptor, Mat& cropped_head_img)
{
  if (drop(fisheye_HOG_descriptor, cropped_head_img, preallocated_leaves_index))
    return preallocated_leaves_index;
  else
    return 0;
}

bool ferns::drop(vector<float> &fisheye_HOG_descriptor, Mat& cropped_head_img, int * leaves_index) {
  for(int i = 0; i < number_of_ferns; i++) {
    int index = 0;
    for (int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;
      index <<= 1;
      if (fern_rules[k].type == 1) {		// HOG
    	//cout << fisheye_HOG_descriptor[fern_rules[k].idxs[0]] << "," << fisheye_HOG_descriptor[fern_rules[k].idxs[1]] << endl;
        if (fisheye_HOG_descriptor[fern_rules[k].idxs[0]] <
        		fisheye_HOG_descriptor[fern_rules[k].idxs[1]]) index++;
      }
      else if (fern_rules[k].type == 2){	// CTC
        Vec3b first_color, second_color, third_color;
        first_color  = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[0], fern_rules[k].idxs[1]);
        second_color = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[2], fern_rules[k].idxs[3]);
        third_color  = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[4], fern_rules[k].idxs[5]);
        if (norm(first_color, second_color, NORM_L1) < norm(second_color, third_color, NORM_L1)) index++;
      }
    }
    leaves_index[i] = index;
    //cout << "  " << bitset<16>(leaves_index[i]);
  }
  return true;
}

int * ferns::drop(float *fisheye_HOG_descriptor, Mat& cropped_head_img)
{
  if (drop(fisheye_HOG_descriptor, cropped_head_img, preallocated_leaves_index))
    return preallocated_leaves_index;
  else
    return 0;
}

bool ferns::drop(float *fisheye_HOG_descriptor, Mat& cropped_head_img, int * leaves_index) {
  for(int i = 0; i < number_of_ferns; i++) {
    int index = 0;
    for (int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;
      index <<= 1;
      if (fern_rules[k].type == 1) {		// HOG
    	//cout << fisheye_HOG_descriptor[fern_rules[k].idxs[0]] << "," << fisheye_HOG_descriptor[fern_rules[k].idxs[1]] << endl;
        if (fisheye_HOG_descriptor[fern_rules[k].idxs[0]] <
        		fisheye_HOG_descriptor[fern_rules[k].idxs[1]]) index++;
      }
      else if (fern_rules[k].type == 2){	// CTC
        Vec3b first_color, second_color, third_color;
        first_color  = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[0], fern_rules[k].idxs[1]);
        second_color = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[2], fern_rules[k].idxs[3]);
        third_color  = cropped_head_img.at<Vec3b>(fern_rules[k].idxs[4], fern_rules[k].idxs[5]);
        if (norm(first_color, second_color, NORM_L1) < norm(second_color, third_color, NORM_L1)) index++;
      }
    }
    leaves_index[i] = index;
    //cout << "  " << bitset<16>(leaves_index[i]);
  }
  return true;
}

// private:
void ferns::alloc(int number_of_ferns, int number_of_tests_per_fern)
{
  this->number_of_ferns = number_of_ferns;
  this->number_of_tests_per_fern = number_of_tests_per_fern;
  number_of_leaves_per_fern = 1 << number_of_tests_per_fern;
  preallocated_leaves_index = new int [number_of_ferns];

  int nb_tests = number_of_ferns * number_of_tests_per_fern;

  fern_rules = new decision_rule[nb_tests];

  /*D_full_images = new int[2 * nb_tests];
  for(int i = 0; i < maximum_number_of_octaves; i++)
    D_aztec_pyramid[i] = new int[2 * nb_tests];*/
}

//void ferns::pick_random_tests(int dx_min, int dx_max, int dy_min, int dy_max, int /*ds_min*/, int /*ds_max*/)
/*{
  for(int i = 0; i < number_of_ferns; i++)
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;

      DX1[k] = dx_min + rand() % (dx_max - dx_min + 1);
      DY1[k] = dy_min + rand() % (dy_max - dy_min + 1);
      DX2[k] = dx_min + rand() % (dx_max - dx_min + 1);
      DY2[k] = dy_min + rand() % (dy_max - dy_min + 1);
      DS1[k] = DS2[k] = 0;
    }

  compute_max_d();
}*/

void ferns::pick_random_tests(int hog_length, int img_width, int img_height, int margin)
{
  for(int i = 0; i < number_of_ferns; i++)
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;

      if (rand() % 100 < HOG_TO_CTC_RATIO) {
        // HOG
        fern_rules[k].type = 1;
        for(int l = 0; l < 2; l++)
          fern_rules[k].idxs[l] = rand() % (hog_length);
        for(int l = 3; l < 6; l++)
          fern_rules[k].idxs[l] = 0;
      }
      else {
        // CTC
        fern_rules[k].type = 2;
        for(int l = 0; l < 6; l=l+2) {
          fern_rules[k].idxs[l] = rand() % (img_width-2*margin) + margin;
          fern_rules[k].idxs[l+1] = rand() % (img_height-2*margin) + margin;
        }
      }
    }
}

/*
void ferns::precompute_D_array(int * D, IplImage * image)
{
  for(int i = 0; i < number_of_ferns; i++)
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;
      D[2 * k]     = DX1[k] + image->widthStep * DY1[k];
      D[2 * k + 1] = DX2[k] + image->widthStep * DY2[k];
    }
}

void ferns::compute_max_d(void)
{
  max_d = 0;

  for(int i = 0; i < number_of_ferns; i++)
    for(int j = 0; j < number_of_tests_per_fern; j++) {
      int k = i * number_of_tests_per_fern + j;

      if (abs(DX1[k]) > max_d) max_d = abs(DX1[k]);
      if (abs(DY1[k]) > max_d) max_d = abs(DY1[k]);
      if (abs(DX2[k]) > max_d) max_d = abs(DX2[k]);
      if (abs(DY2[k]) > max_d) max_d = abs(DY2[k]);
    }
}
*/

void ferns::visualize(string filename, Mat& img, vector<float>& hog_des, int gt_angle, int gt_category) {
	namedWindow(filename);
	Mat base_img(900,1600,CV_8UC3,Scalar(64,64,64));
	Mat resized_img;
	Size resized_size(400,400);
	Point img_origin(80,80);
	Point hog_origin(64,540);
	int scale = resized_size.width/hog_image_size;
	resize(img, resized_img, resized_size, 0, 0, INTER_NEAREST);
	resized_img.copyTo(base_img(Rect(img_origin, resized_size)));
	vector<Point> hog_points;
	Mat hog_mat = draw_hog(hog_des, hog_points);
	hog_mat.copyTo(base_img(Rect(hog_origin, Size(432,120))));
	char angle_txt[50];
	sprintf(angle_txt, "Angle: %3d   , Category: %d", gt_angle, gt_category);
	putText(base_img, angle_txt, Point(40,60), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);

	char buffer[100];
	for (int fern = 0; fern < 1; fern++) {
		for (int leaf = 0; leaf < number_of_tests_per_fern; leaf++) {
			Mat show_img;
			base_img.copyTo(show_img);
			sprintf(buffer, "Fern %d, Leaf %d", fern, leaf);
			putText(show_img, buffer, Point(600,60), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
			int k = fern * number_of_tests_per_fern + leaf;
			if (fern_rules[k].type == 1) {		// HOG
				sprintf(buffer, "HOG");
				putText(show_img, buffer, Point(600,100), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
				int index1 = fern_rules[k].idxs[0];
				int index2 = fern_rules[k].idxs[1];
				sprintf(buffer, "First histogram at %d, value: %.6f", index1, hog_des[index1]);
				putText(show_img, buffer, Point(600,150), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
				sprintf(buffer, "Second histogram at %d, value: %.6f", index2, hog_des[index2]);
				putText(show_img, buffer, Point(600,200), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
				sprintf(buffer, "Decision: %d", hog_des[index1] < hog_des[index2]);
				putText(show_img, buffer, Point(600,350), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
				rectangle(show_img, Rect(hog_origin+hog_points[index1], Size(8,20)), Scalar(0,255,0));
				rectangle(show_img, Rect(hog_origin+hog_points[index2], Size(8,20)), Scalar(0,0,255));
			}
			else if (fern_rules[k].type == 2){	// CTC
				sprintf(buffer, "CTC");
				putText(show_img, buffer, Point(600,100), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
				Point P1(fern_rules[k].idxs[0], fern_rules[k].idxs[1]);
				Point P2(fern_rules[k].idxs[2], fern_rules[k].idxs[3]);
				Point P3(fern_rules[k].idxs[4], fern_rules[k].idxs[5]);
				sprintf(buffer, "First color at  (%2d,%2d), value: [%3d,%3d,%3d]", P1.x, P1.y, img.at<Vec3b>(P1)[0],img.at<Vec3b>(P1)[1],img.at<Vec3b>(P1)[2]);
				putText(show_img, buffer, Point(600,150), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
				sprintf(buffer, "Second color at (%2d,%2d), value: [%3d,%3d,%3d]", P2.x, P2.y, img.at<Vec3b>(P2)[0],img.at<Vec3b>(P2)[1],img.at<Vec3b>(P2)[2]);
				putText(show_img, buffer, Point(600,200), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255), 2);
				sprintf(buffer, "Third color at  (%2d,%2d), value: [%3d,%3d,%3d]", P3.x, P3.y, img.at<Vec3b>(P3)[0],img.at<Vec3b>(P3)[1],img.at<Vec3b>(P3)[2]);
				putText(show_img, buffer, Point(600,250), FONT_HERSHEY_PLAIN, 2, Scalar(255,0,0), 2);
				int norm1 = norm(img.at<Vec3b>(P1), img.at<Vec3b>(P2),NORM_L1);
				int norm2 = norm(img.at<Vec3b>(P2), img.at<Vec3b>(P3),NORM_L1);
				sprintf(buffer, "Norm1: %3d   Norm2: %3d", norm1, norm2);
				putText(show_img, buffer, Point(600,300), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
				sprintf(buffer, "Decision: %d", norm1 < norm2);
				putText(show_img, buffer, Point(600,350), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255), 2);
				rectangle(show_img, Rect(img_origin+scale*P1, Size(20,20)), Scalar(0,255,0));
				rectangle(show_img, Rect(img_origin+scale*P2, Size(20,20)), Scalar(0,0,255));
				rectangle(show_img, Rect(img_origin+scale*P3, Size(20,20)), Scalar(255,0,0));
			}
			imshow(filename, show_img);
			waitKey(0);
		}
	}
}

Mat ferns::draw_hog(vector<float>& hog_des, vector<Point>& hog_points) {
	Mat hog_mat(120,432,CV_8UC3);
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
						Point top_left((col*2*9 + cell_col*9 + bin)*8, (row*2*1 + cell_row)*20);
						hog_points.push_back(top_left);
						uchar pixel_value = hog_des[index]*512;
						//unsigned int pixel_value = index%256;
						//cout << pixel_value << ",";
						Mat pad(20,8,CV_8UC3,Scalar(pixel_value, pixel_value, pixel_value));
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
