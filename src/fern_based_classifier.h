/*
 * fern_based_classifier.h
 *
 *  Created on: Dec 29, 2017
 *      Author: Veerachart Srisamosorn
 */

#ifndef FERN_BASED_CLASSIFIER_H_
#define FERN_BASED_CLASSIFIER_H_

#include <fstream>
using namespace std;

#include "ferns.h"
#include <opencv2/opencv.hpp>

class fern_based_classifier
{
 public:
  fern_based_classifier(char * filename);
  fern_based_classifier(ifstream & f);
  bool correctly_read;

  fern_based_classifier(int number_of_classes,
                              int number_of_ferns, int number_of_tests_per_fern,
                              int hog_img_size = 20, bool paper=false);

  fern_based_classifier(fern_based_classifier * classifier);
  ~fern_based_classifier();

  bool save(char * filename);
  bool save(ofstream & f);

  //! Call this function BEFORE CALLING the train function.
  void reset_leaves_distributions(int prior_number = 1);

  //! You can call this function with different images.
  //! The KEYPOINT CLASSES must be given by the class_index field of the keypoint class.
  void train(vector<vector<float> > &descriptors, vector<cv::Mat> &input_images,
		     vector<int> &ground_truths, int number_of_training_data);

  //! YOU MUST CALL finalize_training() AFTER CALLING train().
  //! IT COMPUTES THE POSTERIOR PROBAS FROM THE NUMBER OF SAMPLES:
  void finalize_training(void);

  //! Used for graph generations:
  void set_number_of_ferns_to_use(int number_of_ferns_to_use);
  int  get_number_of_ferns_to_use(void);

  void print_distributions(void);

  int recognize(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img);
  void recognize(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, double& output_score);
  void recognize_interpolate(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, int& output_angle, float walking_dir=-1.);
  void recognize_interpolate(float *fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, int& output_angle, float walking_dir=-1.);

  void load(ifstream & f);

  ferns * Ferns;

  int number_of_classes;
  short * leaves_counters;
  float * leaves_distributions;
  int step1, step2;
  int * number_of_samples_for_class;
  int prior_number;
  int number_of_ferns_to_use;
  int hog_image_size;
  bool paper_prob;

  double * preallocated_distribution_for_a_keypoint;
};

#endif /* FERN_BASED_CLASSIFIER_H_ */
