#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER:   Taylor Weiss
# DATE CREATED: 07/09/208
# REVISED DATE:
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir
from time import time, sleep

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # DONE: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # DONE: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    # check_command_line_arguments(in_arg)

    # DONE: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
    # check_creating_pet_image_labels(answers_dic)

    # DONE: 4. Define classify_images() function to create the classifier
    # labels with the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    # check_classifying_images(result_dic)

    # DONE: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)
    # check_classifying_labels_as_dogs(result_dic)

    # DONE: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
    # check_calculating_results(result_dic, results_stats_dic)

    # DONE: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch)

    # DONE: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # DONE: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
        str(int(tot_time / 3600)) + ":" +
        str(int(tot_time % 3600 / 60)) + ":" +
        str(round(tot_time % 3600 % 60)))

# TODO: 2.-to-7. Define all the function below. Notice that the input
# paramaters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--dir",
            help="path to the pet image files",
            default="pet_images/")
    parser.add_argument(
            "--arch",
            help="CNN model architecture for image classification",
            choices=["vgg", "alexnet", "resnet"],
            default="vgg")
    parser.add_argument(
            "--dogfile",
            help="text file containing the dog labels",
            default="dognames.txt")
    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    petlabels = {}
    filenames = listdir(image_dir)
    for filename in filenames:
        petlabels[filename] = get_label_from_file_name(filename)
    return petlabels


def get_label_from_file_name(filename):
    """
    Convert a pet image file name into a label.
        1. All lowercase letters
        2. Blank space separating words
        3. Strip whitespace from front and end
    """
    label = ""
    words = filename.lower().split('_')
    for word in words:
        if word.isalpha():
            label += word + ' '
    label = label.strip()
    return label


def classify_images(images_dir, petlable_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    results_dic = {}
    for filename, label in petlable_dic.items():
        classifier_label = classifier(images_dir + filename, model)
        classifier_label = classifier_label.strip().lower()
        results_dic[filename] = [
                label,
                classifier_label,
                1 if is_label_match(label, classifier_label) else 0]
    return results_dic


def is_label_match(pet_label, class_labels):
    """
    Does the pet_label match one of the class_labels?

    class_lables are comma separated. We have a match if the pet_label
    matches from the start to end of one of the class_labels or matches
    from the start of a word to the end of one of the class_lables

    return True for a match, False otherwise
    """
    class_label_parts = class_labels.split(',')
    for part in class_label_parts:
        class_label = part.strip()
        index = class_label.find(pet_label)
        if ((index == 0 and len(pet_label) == len(class_label)) or
            (index > 0 and class_label[index - 1] == ' ' and len(pet_label) + index == len(class_label))):
            return True
    return False


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # read the dognames from the file into a set
    dognames = set()
    with open(dogsfile, 'r') as f:
        line = f.readline()
        while line:
            dognames.add(line.rstrip())
            line = f.readline()

    # update the results_dic with match data
    for image_name, image_data in results_dic.items():
        image_match = 1 if image_data[0] in dognames else 0
        class_match = 1 if image_data[1] in dognames else 0
        results_dic[image_name].extend([image_match, class_match])


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    # tally all the data needed to compute the statistics
    n_images = len(results_dic)
    n_dogs_img = 0              # image a dog [3]
    n_match = 0                 # labels match [2]
    n_correct_dogs = 0          # image a dog [3] and class is a dog [4]
    n_correct_not_dogs = 0      # not a dog ![3] and class is not a dog ![4]
    n_correct_breed = 0         # labels match [2] and is a dog[3]

    for _, result in results_dic.items():
        n_dogs_img += result[3]
        n_match += result[2]
        n_correct_dogs += 1 if result[3] and result[4] else 0
        n_correct_not_dogs += 1 if result[3] == 0 and result[4] == 0 else 0
        n_correct_breed += 1 if result[2] and result[3] else 0

    # compute the statistics
    results_stats = {}

    results_stats['n_images'] = n_images
    results_stats['n_match'] = n_match
    results_stats['n_dogs_img'] = n_dogs_img
    results_stats['n_notdogs_img'] = n_images - n_dogs_img

    results_stats['pct_match'] = \
            n_match * 100.0 / n_images if n_images else 0.0
    results_stats['pct_correct_dogs'] = \
            n_correct_dogs * 100.0 / n_dogs_img if n_dogs_img else 0.0
    results_stats['pct_correct_notdogs'] = \
            n_correct_not_dogs * 100.0 / (n_images - n_dogs_img) if n_images - n_dogs_img else 0.0
    results_stats['pct_correct_breed'] = \
            n_correct_breed * 100.0 / n_dogs_img if n_dogs_img else 0.0

    return results_stats


def print_results(results_dic, results_stats, model):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
   Returns:
           None - simply printing results.
    """
    print("** Results for Architecture", model.upper())
    print()
    print("{:>28} : {:3d}".format("Image Count", results_stats['n_images']))
    print("{:>28} : {:3d}".format("Dog Image Count", results_stats['n_dogs_img']))
    print("{:>28} : {:3d}".format("Not Dog Image Count", results_stats['n_notdogs_img']))
    print()
    print("{:>28} : {:3.1f}".format("Percent Match", results_stats['pct_match']))
    print("{:>28} : {:3.1f}".format("Percent Correct Dogs", results_stats['pct_correct_dogs']))
    print("{:>28} : {:3.1f}".format("Perecent Correct Breed", results_stats['pct_correct_breed']))
    print("{:>28} : {:3.1f}".format("Percent Correct Not Dogs", results_stats['pct_correct_notdogs']))


# Call to main function to run the program
if __name__ == "__main__":
    main()
