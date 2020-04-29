# Introduction
A merge of two different approaches to SLAM: [Direct Sparse Odometry](http://vision.in.tum.de/dso) and [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).
Additionally merged [Stereo-DSO](https://github.com/RonaldSun/VI-Stereo-DSO) to provide capabilities of both Mono and Stereo SLAM.

# Build instructions

Under Windows using vcpkg. The script 'installDependencies.bat' installs all the necessary dependecies.

Additionally Pangolin must be build within the thridparty Folder as well as g2o.

It might be necessary to copy .dll files by hand under windows.

# Open-CL

Optimization within this project utilizes the GPU using OpenCL through openCV's interface. Make sure OpenCV is compiled with the OpenCL option and you have appropriate drivers installed.
Intel should support OpenCL by default, AMD and NVidia might need additional driver setup.

# DBoW

ORB-SLAM2's DBoW is slightly modified and hard-added in this repository.
See [https://github.com/dorian3d/DBow](https://github.com/dorian3d/DBow).

# Run

The project uses gflags for easier flag handling. Use '--help' to see available options.



### 5 License
DSO was developed at the Technical University of Munich and Intel.
The open-source version is licensed under the GNU General Public License
Version 3 (GPLv3).
For commercial purposes, we also offer a professional version, see
[http://vision.in.tum.de/dso](http://vision.in.tum.de/dso) for
details.

Licenses for ORB-SLAM2, DboW are copied within this repository.

