# Assignment Name

Short 1–2 sentence description of what this project does.

## Project Overview

Explain the goal of the assignment.

Example:
This project compares two images using SIFT feature matching. It finds matching keypoints, estimates the geometric transformation between the images, and outputs translation, rotation, scale, and a visualization of the matches.

## Features

- Detects keypoints in two images
- Matches similar points between the images
- Filters bad matches using Lowe's ratio test
- Estimates affine transformation using RANSAC
- Prints translation, rotation, scale, and shear
- Saves an output image with match lines

## Requirements

- Python 3.10+
- OpenCV
- NumPy

Install dependencies:

```bash
py -m pip install opencv-contrib-python numpy