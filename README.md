# Introduction

This project represents the merger of two different approaches related to the processing of depth and mapping of nearby environments, namely, DSO and ORB-SLAM.

ORB-SLAM2 is a real-time SLAM library for Monocular, Stereo and RGB-D cameras that computes the camera trajectory and a sparse 3D reconstruction (in the stereo and RGB-D case with true scale). 

# How to Build Under Windows

To build under windows you need vcpkg and cmake installed in C:/.

We have make.bat which is a script file that we are required to run run that will install most of the dependencies.

## g2o
After the dependencies have been installed by running the bash script file, we must then build the g2o package. We then need to copy the dll that is generated in the g2o folder to the 'bin' folder.

## Sequence Folder

Within the properties of the DSO project, we are required to set the file path to our particular video sequence. the following picture can be used as an example:

"sequenceFolder=C:\Users\user_name\Dcouments\dso_seq1" without the quotation marks.


# Build & run

All that is then left is to build the project.

We can then run our project in Debug mode first to check whether or not there are any errors present.

Assuming that we don not encounter any errors, we can then run the project in Release mode. 

# Open-CL tests

The functions that we are optimizing and running on the iGPU using the OpenCL library are in the file: 'InputTests.cpp'

### 5 License
DSO was developed at the Technical University of Munich and Intel.
The open-source version is licensed under the GNU General Public License
Version 3 (GPLv3).
For commercial purposes, we also offer a professional version, see
[http://vision.in.tum.de/dso](http://vision.in.tum.de/dso) for
details.


