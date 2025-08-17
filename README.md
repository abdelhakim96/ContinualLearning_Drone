# Continual Learning Controllers for Drone

This repository contains Continual Learning controllers for the kinematic and dynamic models of a drone.

## Overview

This project implements several control strategies for a drone, including PID, inverse dynamics, and Deep Neural Network (DNN) based controllers. It also includes a ROS 2 wrapper to control the drone in a simulated environment.

**Author:** Andriy Sarabakha<br />
**Affiliation:** [Nanyang Technological University (NTU)](https://www.ntu.edu.sg), Singapore<br />
**Maintainer:** Andriy Sarabakha, andriy.sarabakha@ntu.edu.sg

**Keywords:** continual learning, deep learning, drone, ros2

This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

This package has been tested with **python3** in **Ubuntu 22.04** and **Ubuntu 20.04**.

### Dependencies

- **pandas** - python data analysis library
  
      pip install pandas

- **torch** - machine learning framework
  
      pip install torch torchvision torchaudio

- (optional) **termcolor** - color formatting in terminal
  
      pip install termcolor

- (optional) **tensorboard** - tool for visualization during the learning
  
      pip install tensorboard

### Clone

Clone the latest version from this repository into your python workspace:

    git clone https://github.com/abdelhakim96/ContinualLearning_Drone.git

## Usage

### Drone Dynamics Simulation:

Run `main_dynamics.py`:

    python python/main_dynamics.py

### Drone Kinematics Simulation:

Run `main_kinematics.py`:

    python python/main_kinematics.py

### ROS 2 Node

To run the ROS 2 node, first build your colcon workspace:

    colcon build

Then source the workspace:

    source install/setup.bash

Finally, run the node:

    ros2 run unicycle_control unicycle_node

You can send velocity commands to the `/cmd_vel` topic and monitor the drone's odometry on the `/odom` topic.
