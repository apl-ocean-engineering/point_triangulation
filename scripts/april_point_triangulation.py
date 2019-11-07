#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import argparse
import cv2
import numpy as np
import copy
import glob

import sys
from os.path import dirname, abspath
from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
from stereoProcessing.point_identification_april import PointIdentificationApril

def calculate_norm_distance(distance):
    """
    Returns array of norm distances between points selected
    """
    norm_distances = []
    for i in range(distance.shape[1]):
        norm_distances.append(np.linalg.norm(distance[0:2, i]))

    return norm_distances


def calculate_distances(pnt4):
    """
    Returns array of distances between corresponding points in order of Clicked
    """
    # Convert to homgenous
    pnt4 /= pnt4[3]
    distance = np.zeros((3, pnt4.shape[1]/2))
    for i in range(0, pnt4.shape[1] - 1, 2):
        distance[0:3, i/2] = np.subtract(pnt4[0:3, i], pnt4[0:3, i+1])

    return distance


def main():
    parser = argparse.ArgumentParser(description="Triangulation of AMP image \
                                     points")
    parser.add_argument("--image_path", help="Path to calibration images",
                    default="/home/tanner/SERDP_images/2apriltag_subset")
    # parser.add_argument("--img1", help="Path to img1", default="/left1.png")
    # parser.add_argument("--img2", help="Path to img2", default="/right1.png")
    parser.add_argument("--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/cfg/calibrationConfig.yaml")
    parser.add_argument("--base_path",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(__file__))) + "/calibration_values/")

    args = parser.parse_args()


    # img_path = args.image_path
    # fname1 = img_path + args.img1
    # fname2 = img_path + args.img2


    # img1 = cv2.imread(fname1)
    # img2 = cv2.imread(fname2)

    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)
    PI = PointIdentificationApril(EI_loader)

    left_path = args.image_path + "/left/"
    right_path = args.image_path + "/right/"
    left_filenames = sorted(glob.glob(left_path + "*.png"))
    right_filenames = sorted(glob.glob(right_path + "*.png"))

    for left_fname, right_fname in zip(left_filenames, right_filenames):
        img1 = cv2.imread(left_fname, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(right_fname, cv2.IMREAD_GRAYSCALE)
        print(left_fname)
        print(right_fname)
        points4D = PI.get_points(img1, img2)
        points4D/=points4D[3]
    print("3D points: ")
    try:
        print(points4D[:3])
    except NameError:
        print "No points created"

    sys.exit()


if __name__ == '__main__':
    """
    Click on points in 2 images IN ORDER. Click on same number of points, and
    output will display 3D distances and norm distance
    """
    main()
