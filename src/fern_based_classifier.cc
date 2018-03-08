/*
 * fern_based_classifier.cc
 *
 *  Created on: Dec 29, 2017
 *      Author: Veerachart Srisamosorn
 */
#include <iostream>
using namespace std;

#include <zlib.h>
#include "fern_based_classifier.h"

fern_based_classifier::fern_based_classifier(char * filename)
{
  ifstream f(filename);

  if (!f.is_open()) {
    cerr << "fern_based_classifier(char * filename): error reading file " << filename << "." << endl;
    correctly_read = false;
    return;
  }

  load(f);

  f.close();
}

fern_based_classifier::fern_based_classifier(ifstream & f)
{
  load(f);
}

fern_based_classifier::fern_based_classifier(int number_of_classes,
                                                         int number_of_ferns, int number_of_tests_per_fern,
                                                         int hog_img_size, bool paper)
{
  this->number_of_classes = number_of_classes;
  hog_image_size = hog_img_size;

  Ferns = new ferns(number_of_ferns, number_of_tests_per_fern,
                    hog_image_size);

  number_of_samples_for_class = new int[number_of_classes];

  int buffer_size = number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern;
  leaves_counters = new short[buffer_size];
  leaves_distributions = new float[buffer_size];
  paper_prob = paper;
  if (paper_prob)
	  reset_leaves_distributions(0);
  else
	  reset_leaves_distributions();
  step1 = number_of_classes;
  step2 = step1 * Ferns->number_of_leaves_per_fern;

  preallocated_distribution_for_a_keypoint = new double[number_of_classes];

  set_number_of_ferns_to_use(-1);
}

fern_based_classifier::fern_based_classifier(fern_based_classifier * classifier) {
	number_of_classes = classifier->number_of_classes;
	hog_image_size = classifier->hog_image_size;
	Ferns = new ferns(classifier->Ferns);
	int buffer_size = number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern;

	leaves_counters = new short[buffer_size];
	for (int i = 0; i < buffer_size; i++)
		leaves_counters[i] = classifier->leaves_counters[i];
	leaves_distributions = new float[buffer_size];
	for (int i = 0; i < buffer_size; i++)
		leaves_distributions[i] = classifier->leaves_distributions[i];
	number_of_samples_for_class = new int[number_of_classes];
	for (int i = 0; i < number_of_classes; i++)
		number_of_samples_for_class[i] = classifier->number_of_samples_for_class[i];
	paper_prob = classifier->paper_prob;
	step1 = number_of_classes;
	step2 = step1 * Ferns->number_of_leaves_per_fern;
	preallocated_distribution_for_a_keypoint = new double[number_of_classes];
	set_number_of_ferns_to_use(-1);
}

fern_based_classifier::~fern_based_classifier()
{
  delete Ferns;
  delete [] leaves_distributions;
  delete [] leaves_counters;
  delete [] preallocated_distribution_for_a_keypoint;
  delete [] number_of_samples_for_class;
}

void fern_based_classifier::load(ifstream & f)
{
  f >> number_of_classes >> hog_image_size >> paper_prob;

  Ferns = new ferns(f);

  if (!Ferns->correctly_read) {
    cerr << ">! [fern_based_classifier::load] Error while reading ferns." << endl;
    correctly_read = false;
    return;
  }

  number_of_samples_for_class = new int[number_of_classes];
  f.read((char *)number_of_samples_for_class, sizeof(int) * number_of_classes);

  int buffer_size = number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern;
  leaves_distributions = new float[buffer_size];
  leaves_counters = new short[buffer_size];

  cout << "> [fern_based_classifier::load] Reading compressed leaves distributions..." << flush;
  int size_of_compressed_buffer, read_buffer_size;
  f >> size_of_compressed_buffer >> read_buffer_size;
  Bytef * compressed_buffer = new Bytef[size_of_compressed_buffer];
  char c; do f.read(&c, 1); while (c != '.');
  f.read((char *)compressed_buffer, size_of_compressed_buffer);
  uLongf uncompressed_buffer_size = buffer_size * sizeof(short);
  (void)uncompress((Bytef*)leaves_counters, &uncompressed_buffer_size, compressed_buffer, size_of_compressed_buffer);
  delete [] compressed_buffer;
  cout << "uncompressed..."<<flush;

  step1 = number_of_classes;
  step2 = step1 * Ferns->number_of_leaves_per_fern;

  finalize_training();

  set_number_of_ferns_to_use(-1);

  preallocated_distribution_for_a_keypoint = new double[number_of_classes];

  correctly_read = true;
  cout << " done." << endl;
  /*for (int i = 0; i < buffer_size; i++) {
	cout << leaves_distributions[i] << ",";
	if (i % step2 == 0)
	  cout << endl;
  }*/
}

bool fern_based_classifier::save(char * filename)
{
  ofstream f(filename);

  if (!f.is_open()) {
    cerr << "> [fern_based_classifier::save] Error while saving file " << filename << "." << endl;

    return false;
  }

  bool result = save(f);

  f.close();

  return result;
}

bool fern_based_classifier::save(ofstream & f)
{
  f << number_of_classes << " " << hog_image_size << " " << int(paper_prob) << endl;

  Ferns->save(f);

  /*char csvname[50];
  for (int i = 0; i < Ferns->number_of_ferns; i++) {
    sprintf(csvname,"leaves_dist%02d.csv",i);
    FILE * csvfile;
    csvfile = fopen(csvname,"w");
    for (int j = 0; j < Ferns->number_of_leaves_per_fern; j++) {
      for (int k = 0; k < number_of_classes-1; k++) {
        fprintf(csvfile,"%.6f,", leaves_distributions[i*step2 + j*step1 + k]);
        //fprintf(csvfile,"%d,", leaves_counters[i*step2 + j*step1 + k]);
      }
      fprintf(csvfile,"%.6f", leaves_distributions[i*step2 + j*step1 + number_of_classes-1]);
      fprintf(csvfile,"\n");
    }
    fclose(csvfile);
  }*/


  f.write((char *)number_of_samples_for_class, sizeof(int) * number_of_classes);

  int buffer_size = number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern;
  //  f.write((char *)leaves_counters, sizeof(short) * buffer_size);

  cout << "> [fern_based_classifier::save] Compressing leaves distributions..." << endl;
  Bytef * compressed_buffer = new Bytef[buffer_size * sizeof(short)];
  uLongf size_of_compressed_buffer = buffer_size * sizeof(short);
  int z_error = compress(compressed_buffer, &size_of_compressed_buffer, (Bytef *)leaves_counters, buffer_size * sizeof(short));
  cout << "> [fern_based_classifier::save] z_error = " << z_error << endl;
  cout << "> [fern_based_classifier::save] size of compressed buffer = " << size_of_compressed_buffer << endl;
  cout << "> [fern_based_classifier::save] Ok. Compression ratio = " << float(buffer_size * sizeof(short)) / size_of_compressed_buffer << "." << endl;
  f << size_of_compressed_buffer << " " << buffer_size << endl;
  char dot('.'); f.write(&dot, 1);
  f.write((char *)compressed_buffer, size_of_compressed_buffer);

  delete [] compressed_buffer;

  return true;
}

void fern_based_classifier::reset_leaves_distributions(int _prior_number)
{
  int buffer_size = number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern;

  prior_number = _prior_number;

  for(int i = 0; i < buffer_size; i++)
    leaves_counters[i] = short(prior_number);
  for(int i = 0; i < number_of_classes; i++)
    number_of_samples_for_class[i] = 0;
}

void fern_based_classifier::train(vector<vector<float> > &descriptors, vector<cv::Mat> &input_images, vector<int> &ground_truths, int number_of_training_data){
  for (int i = 0; i < number_of_training_data; i++) {
    int * leaves_index = Ferns->drop(descriptors[i], input_images[i]);		// TODO from descriptor's size
    if (leaves_index != 0) {
      number_of_samples_for_class[ground_truths[i]]++;
      for(int k = 0; k < Ferns->number_of_ferns; k++) {
        leaves_counters[k * step2 + leaves_index[k] * step1 + ground_truths[i]]++;
      }
    }
    /*cout << i << "\t";
    int sum = 0;
    for (int x = 0; x < number_of_classes * Ferns->number_of_ferns * Ferns->number_of_leaves_per_fern; x++){
    	sum += leaves_counters[x];
    	//cout << leaves_counters[x] << ",";
    }
    cout << sum << endl;*/
  }
}

void fern_based_classifier::finalize_training(void)
{

#pragma omp parallel for
  for(int i = 0; i < Ferns->number_of_ferns; i++) {
	if (paper_prob) {
	  for(int k = 0; k < number_of_classes; k++) {
	    for (int j = 0; j < Ferns->number_of_leaves_per_fern; j++) {
		  leaves_distributions[i*step2 + j*step1 + k] = float( log( double(leaves_counters[i*step2 + j*step1 + k] + 1)/ double(number_of_samples_for_class[k] + Ferns->number_of_leaves_per_fern) ) );
	    }
	  }
	}
	else {
		  double * number_of_samples_for_this_leaf = new double[Ferns->number_of_leaves_per_fern];
		  memset(number_of_samples_for_this_leaf,0,sizeof(double)*Ferns->number_of_leaves_per_fern);

		double number_of_samples_for_this_fern = 0.;
		for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
		  for(int k = 0; k < number_of_classes; k++)
		number_of_samples_for_this_fern +=
		  double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);

		//cout << number_of_samples_for_this_fern << endl;
		for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++) {
		  for(int k = 0; k < number_of_classes; k++) {
		number_of_samples_for_this_leaf[j] +=
		  double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);
		  }
		}

		for(int k = 0; k < number_of_classes; k++) {
		  double sum = 0.;
		  for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
		sum += double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k]);

		  for(int j = 0; j < Ferns->number_of_leaves_per_fern; j++)
		leaves_distributions[i * step2 + j * step1 + k] =
		  float( log( double(leaves_counters[i * step2 + j * step1 + k]) / double(number_of_samples_for_class[k])
				  / sum
				  / (number_of_samples_for_this_leaf[j] / number_of_samples_for_this_fern) ) );
				//float( log( double(leaves_counters[i * step2 + j * step1 + k] + 1) / double(number_of_samples_for_class[k] + Ferns->number_of_leaves_per_fern)));
		}

		delete [] number_of_samples_for_this_leaf;
	}
  }
  //char filename[] = "train_data";
  //save(filename);
}

void fern_based_classifier::set_number_of_ferns_to_use(int _number_of_ferns_to_use)
{
  number_of_ferns_to_use = _number_of_ferns_to_use;
}

int  fern_based_classifier::get_number_of_ferns_to_use(void)
{
  if (number_of_ferns_to_use < 1)
    return Ferns->number_of_ferns;
  else
    return min(Ferns->number_of_ferns, number_of_ferns_to_use);
}

void fern_based_classifier::print_distributions(void)
{
	for (int k = 0; k < 32; k++)
		cout << leaves_distributions[k] << ",";
	cout << endl;
}

int fern_based_classifier::recognize(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img) {
	int * leaves_index = Ferns->drop(fisheye_HOG_descriptor, cropped_head_img);

	if (leaves_index == 0) return -1;

	double * distribution = preallocated_distribution_for_a_keypoint;

	for (int i = 0; i < number_of_classes; i++)
		distribution[i] = 0;

	const int nb_ferns = get_number_of_ferns_to_use();
	for (int i = 0; i < nb_ferns; i++) {
		float * ld = leaves_distributions + i*step2 + leaves_index[i]*step1;
		for (int k = 0; k < number_of_classes; k++)
			distribution[k] += ld[k];
	}

	int class_index = 0;
	double class_score = distribution[0];
	for (int k = 0; k < number_of_classes; k++) {
		//cout << distribution[k] << ",";
		if (distribution[k] > class_score) {
			class_index = k;
			class_score = distribution[k];
		}
	}

	return class_index;
}

void fern_based_classifier::recognize(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, double& output_score) {
	int * leaves_index = Ferns->drop(fisheye_HOG_descriptor, cropped_head_img);

		if (leaves_index == 0) return;

		double * distribution = preallocated_distribution_for_a_keypoint;

		for (int i = 0; i < number_of_classes; i++)
			distribution[i] = 0;

		const int nb_ferns = get_number_of_ferns_to_use();
		for (int i = 0; i < nb_ferns; i++) {
			float * ld = leaves_distributions + i*step2 + leaves_index[i]*step1;
			for (int k = 0; k < number_of_classes; k++)
				distribution[k] += ld[k];
		}

		int class_index = 0;
		double class_score = distribution[0];
		for (int k = 0; k < number_of_classes; k++) {
			//cout << distribution[k] << ",";
			if (distribution[k] > class_score) {
				class_index = k;
				class_score = distribution[k];
			}
		}
		//cout << endl;

		output_class = class_index;
		output_score = class_score;
}

void fern_based_classifier::recognize_interpolate(vector<float> &fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, int& output_angle, float walking_dir) {
	int * leaves_index = Ferns->drop(fisheye_HOG_descriptor, cropped_head_img);
	float class_representations[] = {0, 45, 90, 135, 180, 225, 270, 315};
	float walking_probs[number_of_classes];
	if (walking_dir >= 0) {
		float sigma = 30.;
		for (int i = 0; i < number_of_classes; i++) {
			if (abs(class_representations[i] - walking_dir) > 90. && abs(class_representations[i] - walking_dir) < 270.)
				walking_probs[i] = -100;			// Just reduce the chance
			else {
				float diff = abs(class_representations[i] - walking_dir);
				if (diff > 270.)
					diff = 360. - diff;
				walking_probs[i] = -pow(diff, 2)/(2*sigma*sigma) - 0.5*log(2*CV_PI*sigma);		// Gaussian
			}
		}
	}
	else {
		for (int i = 0; i < number_of_classes; i++)
			walking_probs[i] = 0;
	}

	if (leaves_index == 0) return;

	double * distribution = preallocated_distribution_for_a_keypoint;

	for (int i = 0; i < number_of_classes; i++)
		distribution[i] = 0;

	const int nb_ferns = get_number_of_ferns_to_use();
	for (int i = 0; i < nb_ferns; i++) {
		float * ld = leaves_distributions + i*step2 + leaves_index[i]*step1;
		for (int k = 0; k < number_of_classes; k++)
			distribution[k] += ld[k];
	}

	int class_index = 0;
	double class_score = distribution[0] + walking_probs[0];
	for (int k = 0; k < number_of_classes; k++) {
		//cout << distribution[k] << ",";
		distribution[k] += walking_probs[k];				// Apply walking probability
		//cout << distribution[k] << endl;
		if (distribution[k] > class_score) {
			class_index = k;
			class_score = distribution[k];
		}
	}
	//cout << endl;

	output_class = class_index;
	if (output_class == 8) {		// Not a head
		output_angle = -1;
		return;
	}

	int low, high;
	float low_representation, high_representation;
	if (class_index-1 < 0) {
		low = (number_of_classes == 9 ? number_of_classes-1 : number_of_classes) - 1;
		low_representation = 360-class_representations[low];
	}
	else {
		low = class_index-1;
		low_representation = class_representations[low];
	}
	if (class_index+1 >= (number_of_classes == 9 ? number_of_classes-1 : number_of_classes)) {
		high = 0;
		high_representation = 360+class_representations[high];
	}
	else {
		high = class_index+1;
		high_representation = class_representations[high];
	}

	//double estimate = exp(class_score)*class_representations[class_index] + exp(distribution[low])*low_representation + exp(distribution[high])*high_representation;
	//double sum = exp(class_score) + exp(distribution[low]) + exp(distribution[high]);
	double estimate = class_representations[class_index] + exp(distribution[low]-class_score)*low_representation + exp(distribution[high]-class_score)*high_representation;
	double sum = 1 + exp(distribution[low]-class_score) + exp(distribution[high]-class_score);
	//double estimate2 = class_representations[class_index] + exp(distribution[low]-class_score)*low_representation + exp(distribution[high]-class_score)*high_representation;
	//double sum2 = 1 + exp(distribution[low]-class_score) + exp(distribution[high]-class_score);

	//cout << " " << output_class << ": " << distribution[low] << "," << class_score << "," << distribution[high] << "->" << estimate << "---" << sum << "=" << estimate/sum << endl;

	output_angle = int(round(estimate/sum));
	if (output_angle > 360)
		output_angle -= 360;
	else if (output_angle < 0)
		output_angle += 360;
}

void fern_based_classifier::recognize_interpolate(float *fisheye_HOG_descriptor, cv::Mat& cropped_head_img, int& output_class, int& output_angle, float walking_dir) {
	int * leaves_index = Ferns->drop(fisheye_HOG_descriptor, cropped_head_img);
	float class_representations[] = {0, 45, 90, 135, 180, 225, 270, 315};
	float walking_probs[number_of_classes];
	if (walking_dir >= 0) {
		float sigma = 30.;
		for (int i = 0; i < number_of_classes; i++) {
			if (abs(class_representations[i] - walking_dir) > 90. && abs(class_representations[i] - walking_dir) < 270.)
				walking_probs[i] = -100;			// Just reduce the chance
			else
				walking_probs[i] = -pow(class_representations[i] - walking_dir, 2)/(2*sigma*sigma) - 0.5*log(2*CV_PI*sigma);		// Gaussian
		}
	}
	else {
		for (int i = 0; i < number_of_classes; i++)
			walking_probs[i] = 0;
	}

	if (leaves_index == 0) return;

	double * distribution = preallocated_distribution_for_a_keypoint;

	for (int i = 0; i < number_of_classes; i++)
		distribution[i] = 0;

	const int nb_ferns = get_number_of_ferns_to_use();
	for (int i = 0; i < nb_ferns; i++) {
		float * ld = leaves_distributions + i*step2 + leaves_index[i]*step1;
		for (int k = 0; k < number_of_classes; k++)
			distribution[k] += ld[k];
	}

	int class_index = 0;
	double class_score = distribution[0];
	for (int k = 0; k < number_of_classes; k++) {
		//cout << distribution[k] << ",";
		distribution[k] += walking_probs[k];				// Apply walking probability
		//cout << distribution[k] << endl;
		if (distribution[k] > class_score) {
			class_index = k;
			class_score = distribution[k];
		}
	}
	//cout << endl;

	output_class = class_index;
	if (output_class == 8) {		// Not a head
		output_angle = -1;
		return;
	}

	int low, high;
	float low_representation, high_representation;
	if (class_index-1 < 0) {
		low = (number_of_classes == 9 ? number_of_classes-1 : number_of_classes) - 1;
		low_representation = 360-class_representations[low];
	}
	else {
		low = class_index-1;
		low_representation = class_representations[low];
	}
	if (class_index+1 >= (number_of_classes == 9 ? number_of_classes-1 : number_of_classes)) {
		high = 0;
		high_representation = 360+class_representations[high];
	}
	else {
		high = class_index+1;
		high_representation = class_representations[high];
	}

	//double estimate = exp(class_score)*class_representations[class_index] + exp(distribution[low])*low_representation + exp(distribution[high])*high_representation;
	//double sum = exp(class_score) + exp(distribution[low]) + exp(distribution[high]);
	double estimate = class_representations[class_index] + exp(distribution[low]-class_score)*low_representation + exp(distribution[high]-class_score)*high_representation;
	double sum = 1 + exp(distribution[low]-class_score) + exp(distribution[high]-class_score);
	//double estimate2 = class_representations[class_index] + exp(distribution[low]-class_score)*low_representation + exp(distribution[high]-class_score)*high_representation;
	//double sum2 = 1 + exp(distribution[low]-class_score) + exp(distribution[high]-class_score);

	//cout << " " << output_class << ": " << distribution[low] << "," << class_score << "," << distribution[high] << "->" << estimate << "---" << sum << "=" << estimate/sum << endl;

	output_angle = int(round(estimate/sum));
	if (output_angle > 360)
		output_angle -= 360;
	else if (output_angle < 0)
		output_angle += 360;
}
