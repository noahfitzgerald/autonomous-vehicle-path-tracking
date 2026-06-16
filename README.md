# Autonomous Vehicle Path Tracking

Selected thesis code from my Mechanical Engineering thesis at Cal Poly, San Luis Obispo.

This repository contains cleaned, public-facing Python source code for lane-based path planning and steering controller comparison on an autonomous ground vehicle platform. The work focused on converting detected lane information into local path waypoints, then comparing Stanley and Model Predictive Control steering behavior against recorded vehicle steering data.

## Overview

The thesis workflow used recorded vehicle data, camera/lane-detection outputs, and ROS/Foxglove data streams to evaluate autonomous vehicle path-tracking behavior.

The main workflow was:

1. Process lane mask data from recorded driving runs
2. Convert image-space lane detections into vehicle-frame surface coordinates
3. Estimate local road direction and generate path waypoints
4. Apply steering controllers to the generated path
5. Compare predicted steering commands against recorded steering commands

## Repository Contents

```text
src/
  lane_path_planner.py
  stanley_controller.py
  mpc_controller.py

requirements.txt
README.md
LICENSE
```

## Source Files

### `lane_path_planner.py`

Processes lane mask outputs and generates local path data for controller evaluation.

Key functionality includes:

* Filtering lane detections for left and right lane lines
* Transforming image pixel coordinates into ground-plane surface coordinates using a homography
* Estimating road direction from detected lane geometry
* Generating local path waypoints for controller input
* Providing plotting helpers for visualizing lane extraction and waypoint generation

### `stanley_controller.py`

Implements a Stanley-style steering controller.

Key functionality includes:

* Converting quaternion orientation data to yaw angle
* Parsing recorded path data from ROS/Foxglove-style message structures
* Computing yaw error, cross-track error, yaw-rate damping, and steering smoothing terms
* Applying steering-angle limits
* Returning predicted steering commands for path-tracking comparison

### `mpc_controller.py`

Implements a simplified Model Predictive Controller for steering command prediction.

Key functionality includes:

* Generating local waypoints from path-planner outputs
* Predicting vehicle motion using a kinematic bicycle model
* Defining an MPC cost function with position error, heading error, steering effort, and steering-change penalties
* Solving constrained steering optimization using SLSQP
* Returning the first optimized steering command from the prediction horizon

## Methods

### Lane-Based Path Planning

The path planner uses lane mask data to identify left and right lane boundaries. Detected lane pixels are transformed into ground-plane coordinates, then a local road direction is estimated from the lane geometry. The planner outputs a waypoint containing local position and heading information for use by downstream steering controllers.

### Stanley Controller

The Stanley controller computes steering commands using a combination of heading error and cross-track error. Additional terms were included for yaw-rate damping and steering smoothing. Steering output is clipped to the vehicle steering limit.

### Model Predictive Control

The MPC controller predicts future vehicle states using a kinematic bicycle model. It optimizes a short steering sequence to minimize a weighted cost function based on position tracking error, heading error, steering effort, and steering smoothness. The first steering input from the optimized sequence is used as the predicted steering command.

## External Tools and Platform References

This repository contains selected thesis scripts written by Noah Fitzgerald for lane-based path planning and steering controller comparison.

The thesis workflow used external robotics/data infrastructure developed by William Mx, including:

* foxflow: Used for importing Foxglove Cloud / ROS recording data into Jupyter and Colab workflows.
  https://github.com/william-mx/foxflow

* ros2_pydata: Used or referenced for converting ROS2 message data into Python-native / NumPy-friendly formats.
  https://github.com/william-mx/ros2_pydata

* mxck_interfaces: Used as part of the MXCK platform’s ROS2 custom service/message infrastructure.
  https://github.com/william-mx/mxck_interfaces

* mxck_ws: Supporting ROS workspace for the MXcarkit platform.
  https://github.com/william-mx/mxck_ws

* MXcarkit: Vehicle/platform resources used as supporting infrastructure for the thesis work.
  https://github.com/william-mx/MXcarkit

These external repositories are credited as dependencies, references, or platform infrastructure. They are not claimed as original work in this repository and remain governed by their own licenses.

## Data and Privacy Notes

Raw driving recordings, Foxglove data, local Google Drive paths, API keys, and large exported datasets are not included in this repository.

This repository is intended as a cleaned public-facing version of selected thesis code. It is not a full reproduction of the original private thesis workspace.

## Requirements

Core Python packages used by the source code include:

```text
numpy
pandas
scipy
opencv-python
pyyaml
matplotlib
scikit-learn
```

## Status

This repository is a selected-code thesis archive and portfolio reference. The scripts are intended to document the path-planning and controller-comparison workflow rather than serve as a fully packaged production library.

## Author

Noah Fitzgerald
Mechanical Engineering Thesis Project
California Polytechnic State University, San Luis Obispo

## License

This repository is licensed under the MIT License for code written by Noah Fitzgerald.

External tools and platform repositories referenced in this project remain governed by their own licenses.
